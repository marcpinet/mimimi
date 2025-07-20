import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
import matplotlib.pyplot as plt
from torch.nn import functional as F
import pickle
import random

from model import ComplexAnomalyDetector
from dataset import MIMIIDataset
from config import Config
from utils import find_youden_threshold


def kl_divergence_loss(logits, target_weights):
    log_probs = F.log_softmax(logits, dim=1)
    if log_probs.shape != target_weights.shape:
        raise ValueError("Shape mismatch between log_probs and target_weights")
    return -torch.sum(target_weights * log_probs, dim=1).mean()


def create_custom_collate_fn(num_classes, mixup_alpha):
    def custom_collate_fn(batch):
        if not batch:
            return torch.tensor([]), torch.tensor([])

        data, machine_id_tensors, _ = zip(*batch)

        real_parts = torch.stack([d[0] for d in data], 0)
        imag_parts = torch.stack([d[1] for d in data], 0)

        machine_ids = [mid.item() for mid in machine_id_tensors]

        batch_size = real_parts.size(0)

        spectrograms_by_id = {i: [] for i in range(num_classes)}
        for i in range(batch_size):
            mid = machine_ids[i]
            if mid < num_classes:
                spectrograms_by_id[mid].append((real_parts[i], imag_parts[i]))

        mixed_reals_list = []
        mixed_imags_list = []
        targets_list = []

        for _ in range(batch_size):
            target_weights = np.random.dirichlet([mixup_alpha] * num_classes)

            mixed_real = torch.zeros_like(real_parts[0])
            mixed_imag = torch.zeros_like(imag_parts[0])

            for class_idx in range(num_classes):
                weight = target_weights[class_idx]
                if weight > 1e-8 and spectrograms_by_id[class_idx]:
                    chosen_real, chosen_imag = random.choice(spectrograms_by_id[class_idx])

                    mixed_real += weight * chosen_real
                    mixed_imag += weight * chosen_imag

            mixed_reals_list.append(mixed_real)
            mixed_imags_list.append(mixed_imag)
            targets_list.append(torch.from_numpy(target_weights).float())

        final_mixed_reals = torch.stack(mixed_reals_list, 0)
        final_mixed_imags = torch.stack(mixed_imags_list, 0)
        final_targets = torch.stack(targets_list, 0)

        mixed_data = (final_mixed_reals, final_mixed_imags)

        return mixed_data, final_targets
    
    return custom_collate_fn


class Trainer:
    def __init__(self, machine_type, config=None):
        self.config = config or Config()
        self.machine_type = machine_type
        self.device = torch.device(self.config.DEVICE)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
            'train_f1': [], 'val_f1': [], 'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [], 'anomaly_auroc': [], 'anomaly_f1': [],
            'anomaly_precision': [], 'anomaly_recall': []
        }
        
        self.files_by_id = {}
        self.preloaded_data = {}
        
        self.setup_data()
        self.setup_model()
        self.setup_metrics()
        
    def setup_metrics(self):
        self.train_metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device),
            'precision': Precision(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'recall': Recall(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'f1': F1Score(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device)
        }
        
        self.val_metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device),
            'precision': Precision(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'recall': Recall(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'f1': F1Score(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device)
        }
        
        self.anomaly_metrics = {
            'auroc': AUROC(task='binary').to(self.device),
            'precision': Precision(task='binary').to(self.device),
            'recall': Recall(task='binary').to(self.device),
            'f1': F1Score(task='binary').to(self.device),
        }
        
    def reset_metrics(self, metric_dict):
        for metric in metric_dict.values():
            metric.reset()
            
    def setup_data(self):
        train_dataset = MIMIIDataset(
            self.config.DATA_PATH, self.machine_type, split='train',
            sr=self.config.AUDIO_SR, n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH, n_frames=self.config.N_FRAMES,
            augment=self.config.AUGMENT,
            normalize=self.config.NORMALIZE
        )
        
        if self.config.TRAIN_NORMAL_ONLY:
            normal_indices = [i for i, label in enumerate(train_dataset.labels) if label == 0]
            train_dataset.files = [train_dataset.files[i] for i in normal_indices]
            train_dataset.labels = [train_dataset.labels[i] for i in normal_indices]
            train_dataset.machine_ids = [train_dataset.machine_ids[i] for i in normal_indices]
        
        self.num_classes = train_dataset.num_machine_ids
        print(f"Training with {len(train_dataset)} normal samples from {self.num_classes} machine IDs")
        
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        test_dataset = MIMIIDataset(
            self.config.DATA_PATH, self.machine_type, split='test',
            sr=self.config.AUDIO_SR, n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH, n_frames=self.config.N_FRAMES,
            augment=False,
            normalize=self.config.NORMALIZE
        )
        
        def simple_collate_fn(batch):
            data, machine_id, label = zip(*batch)
            real_parts, imag_parts = [], []
            for d in data:
                real_part, imag_part = d
                real_parts.append(real_part)
                imag_parts.append(imag_part)
            real_batch = torch.stack(real_parts, 0)
            imag_batch = torch.stack(imag_parts, 0)
            machine_id_batch = torch.stack(machine_id, 0)
            label_batch = torch.stack(label, 0)
            return (real_batch, imag_batch), machine_id_batch, label_batch
        
        func = create_custom_collate_fn(self.num_classes, self.config.MIXUP_ALPHA) if self.config.USE_MIXUP else simple_collate_fn
        
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            collate_fn=func,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            collate_fn=simple_collate_fn, 
            num_workers=4
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            collate_fn=simple_collate_fn, 
            num_workers=4
        )

    def setup_model(self):
        self.model = ComplexAnomalyDetector(num_classes=self.num_classes)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        self.reset_metrics(self.train_metrics)

        for batch in tqdm(self.train_loader, desc='Training', leave=False):
            self.optimizer.zero_grad()
            
            if self.config.USE_MIXUP:
                data, targets = batch
                real_batch, imag_batch = data
                real_batch = real_batch.to(self.device)
                imag_batch = imag_batch.to(self.device)
                targets = targets.to(self.device)
                
                logits_magnitude, logits_complex, logits_total = self.model((real_batch, imag_batch))
                
                loss_magnitude = kl_divergence_loss(logits_magnitude, targets)
                loss_complex = kl_divergence_loss(logits_complex, targets)
                loss_total = kl_divergence_loss(logits_total, targets)
                
                loss = loss_total + loss_magnitude + loss_complex
                
                true_labels = torch.argmax(targets, dim=1)
            
            else:
                data, machine_ids, _ = batch
                real_batch, imag_batch = data
                real_batch = real_batch.to(self.device)
                imag_batch = imag_batch.to(self.device)
                machine_ids = machine_ids.to(self.device).squeeze()
                
                logits_magnitude, logits_complex, logits_total = self.model((real_batch, imag_batch))
                
                loss_magnitude = self.criterion(logits_magnitude, machine_ids)
                loss_complex = self.criterion(logits_complex, machine_ids)
                loss_total = self.criterion(logits_total, machine_ids)

                loss = loss_total + loss_magnitude + loss_complex
                
                true_labels = machine_ids
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits_total, dim=1)
            self.train_metrics['accuracy'].update(predictions, true_labels)
            self.train_metrics['precision'].update(predictions, true_labels)
            self.train_metrics['recall'].update(predictions, true_labels)
            self.train_metrics['f1'].update(predictions, true_labels)

        avg_train_loss = total_loss / len(self.train_loader)
        train_accuracy = self.train_metrics['accuracy'].compute().item()
        train_precision = self.train_metrics['precision'].compute().item()
        train_recall = self.train_metrics['recall'].compute().item()
        train_f1 = self.train_metrics['f1'].compute().item()
        
        self.metrics_history['train_loss'].append(avg_train_loss)
        self.metrics_history['train_accuracy'].append(train_accuracy)
        self.metrics_history['train_precision'].append(train_precision)
        self.metrics_history['train_recall'].append(train_recall)
        self.metrics_history['train_f1'].append(train_f1)
        
        return avg_train_loss, train_accuracy, train_precision, train_recall, train_f1
    
    def validate(self):
        self.model.eval()
        total_loss, num_batches = 0, 0
        self.reset_metrics(self.val_metrics)

        with torch.no_grad():
            for data, machine_id, _ in tqdm(self.val_loader, desc='Validation', leave=False):
                real_batch, imag_batch = data
                real_batch, imag_batch = real_batch.to(self.device), imag_batch.to(self.device)
                machine_id = machine_id.to(self.device).squeeze()
                
                logits_magnitude, logits_complex, logits_total = self.model((real_batch, imag_batch))
                loss = F.cross_entropy(logits_magnitude, machine_id) + \
                       F.cross_entropy(logits_complex, machine_id) + \
                       F.cross_entropy(logits_total, machine_id)

                total_loss += loss.item()
                num_batches += 1
                
                predictions = torch.argmax(logits_total, dim=1)
                self.val_metrics['accuracy'].update(predictions, machine_id)
                self.val_metrics['precision'].update(predictions, machine_id)
                self.val_metrics['recall'].update(predictions, machine_id)
                self.val_metrics['f1'].update(predictions, machine_id)

        val_loss = total_loss / num_batches
        val_accuracy = self.val_metrics['accuracy'].compute().item()
        val_precision = self.val_metrics['precision'].compute().item()
        val_recall = self.val_metrics['recall'].compute().item()
        val_f1 = self.val_metrics['f1'].compute().item()

        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_accuracy'].append(val_accuracy)
        self.metrics_history['val_precision'].append(val_precision)
        self.metrics_history['val_recall'].append(val_recall)
        self.metrics_history['val_f1'].append(val_f1)

        return val_loss, val_accuracy, val_precision, val_recall, val_f1
    
    def evaluate_anomaly_detection(self):
        self.model.eval()
        all_scores, all_labels, all_machine_ids = [], [], []
        self.reset_metrics(self.anomaly_metrics)
        
        with torch.no_grad():
            for data, machine_id, label in tqdm(self.test_loader, desc='Evaluation', leave=False):
                real_batch, imag_batch = data
                real_batch, imag_batch = real_batch.to(self.device), imag_batch.to(self.device)
                machine_id, label = machine_id.to(self.device).squeeze(), label.squeeze()
                
                scores = self.model.get_anomaly_score((real_batch, imag_batch), machine_id)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(label.numpy())
                all_machine_ids.extend(machine_id.cpu().numpy())
        
        all_scores, all_labels, all_machine_ids = np.array(all_scores), np.array(all_labels), np.array(all_machine_ids)
        scores_tensor, labels_tensor = torch.tensor(all_scores).to(self.device), torch.tensor(all_labels).to(self.device)
        
        self.anomaly_metrics['auroc'].update(scores_tensor, labels_tensor)
        threshold = find_youden_threshold(scores_tensor.cpu().numpy(), labels_tensor.cpu().numpy())
        binary_preds = (scores_tensor > threshold).int()
        self.anomaly_metrics['precision'].update(binary_preds, labels_tensor)
        self.anomaly_metrics['recall'].update(binary_preds, labels_tensor)
        self.anomaly_metrics['f1'].update(binary_preds, labels_tensor)
        
        anomaly_auroc = self.anomaly_metrics['auroc'].compute().item()
        anomaly_precision = self.anomaly_metrics['precision'].compute().item()
        anomaly_recall = self.anomaly_metrics['recall'].compute().item()
        anomaly_f1 = self.anomaly_metrics['f1'].compute().item()
        
        self.metrics_history['anomaly_auroc'].append(anomaly_auroc)
        self.metrics_history['anomaly_precision'].append(anomaly_precision)
        self.metrics_history['anomaly_recall'].append(anomaly_recall)
        self.metrics_history['anomaly_f1'].append(anomaly_f1)
        
        overall_auc = roc_auc_score(all_labels, all_scores)
        
        machine_id_aucs = {}
        for mid in np.unique(all_machine_ids):
            mask = all_machine_ids == mid
            if len(np.unique(all_labels[mask])) > 1:
                machine_id_aucs[int(mid)] = roc_auc_score(all_labels[mask], all_scores[mask])
        
        return overall_auc, machine_id_aucs, anomaly_auroc, anomaly_precision, anomaly_recall, anomaly_f1
    
    def save_metrics_plots(self, save_dir):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, self.metrics_history['train_accuracy'], 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], 'r-', label='Val Accuracy')
        axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)
        
        axes[0, 2].plot(epochs, self.metrics_history['train_f1'], 'b-', label='Train F1')
        axes[0, 2].plot(epochs, self.metrics_history['val_f1'], 'r-', label='Val F1')
        axes[0, 2].set_title('F1 Score'); axes[0, 2].legend(); axes[0, 2].grid(True)
        
        axes[1, 0].plot(epochs, self.metrics_history['train_precision'], 'b-', label='Train Precision')
        axes[1, 0].plot(epochs, self.metrics_history['val_precision'], 'r-', label='Val Precision')
        axes[1, 0].set_title('Precision'); axes[1, 0].legend(); axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs, self.metrics_history['train_recall'], 'b-', label='Train Recall')
        axes[1, 1].plot(epochs, self.metrics_history['val_recall'], 'r-', label='Val Recall')
        axes[1, 1].set_title('Recall'); axes[1, 1].legend(); axes[1, 1].grid(True)
        
        if self.metrics_history['anomaly_auroc']:
            axes[1, 2].bar(['AUROC', 'Precision', 'Recall', 'F1'], 
                          [self.metrics_history['anomaly_auroc'][-1], self.metrics_history['anomaly_precision'][-1],
                           self.metrics_history['anomaly_recall'][-1], self.metrics_history['anomaly_f1'][-1]])
            axes[1, 2].set_title('Final Anomaly Detection Metrics'); axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{self.machine_type}_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(save_dir / f'{self.machine_type}_metrics_history.pkl', 'wb') as f:
            pickle.dump(self.metrics_history, f)
    
    def train(self):
        print(f"Training {self.machine_type} model with {self.num_classes} machine IDs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        save_dir = Path(self.config.SAVE_DIR)
        save_dir.mkdir(exist_ok=True)
        
        best_auc = -1.0
        
        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc, _, _, _ = self.train_epoch()
            val_loss, val_acc, _, _, _ = self.validate()

            current_auc, _, _, _, _, _ = self.evaluate_anomaly_detection()
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} - Time: {time.time() - start_time:.2f}s")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"Current Anomaly AUC on test set: {current_auc:.4f} (Best so far: {best_auc:.4f})")
            
            if current_auc > best_auc:
                best_auc = current_auc
                self.patience_counter = 0
                model_path = save_dir / f'{self.machine_type}_best_model.pth'
                torch.save({
                    'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'anomaly_auc': best_auc,
                    'num_classes': self.num_classes,
                    'metrics_history': self.metrics_history
                }, model_path)
                print(f"New best model saved with AUC: {best_auc:.4f} at path: {model_path}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.PATIENCE:
                print(f"Early stopping after {epoch+1} epochs due to no AUC improvement.")
                break
                
            print("-" * 50)
        
        print("Training completed!")
        print("Loading best model for final evaluation...")
        
        best_model_path = save_dir / f'{self.machine_type}_best_model.pth'
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Evaluating final anomaly detection performance on the best model...")
        overall_auc, machine_id_aucs, anomaly_auroc, anomaly_precision, anomaly_recall, anomaly_f1 = self.evaluate_anomaly_detection()
        
        print(f"Overall AUC: {overall_auc:.4f}")
        print(f"Anomaly Detection - AUROC: {anomaly_auroc:.4f}, Precision: {anomaly_precision:.4f}, Recall: {anomaly_recall:.4f}, F1: {anomaly_f1:.4f}")
        print("Machine ID AUCs:", {f"ID {mid}": f"{auc:.4f}" for mid, auc in machine_id_aucs.items()})
        
        self.save_metrics_plots(save_dir)
        
        results = {
            'machine_type': self.machine_type, 'overall_auc': float(overall_auc),
            'machine_id_aucs': {str(k): float(v) for k, v in machine_id_aucs.items()},
            'num_classes': int(self.num_classes), 'anomaly_auroc': float(anomaly_auroc),
            'anomaly_precision': float(anomaly_precision), 'anomaly_recall': float(anomaly_recall),
            'anomaly_f1': float(anomaly_f1), 'metrics_history': self.metrics_history
        }
        
        results_path = save_dir / f'{self.machine_type}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main(config=None):
    if config is None:
        config = Config()

    all_results = {}
    
    for machine_type in config.MACHINE_TYPES:
        print(f"\n{'='*60}\nTraining model for {machine_type}\n{'='*60}")
        trainer = Trainer(machine_type, config)
        results = trainer.train()
        all_results[machine_type] = results
        print(f"\nCompleted training for {machine_type}\nAUC: {results['overall_auc']:.4f}")
    
    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    
    total_auc = 0
    for machine_type, results in all_results.items():
        auc = results['overall_auc']
        total_auc += auc
        print(f"{machine_type}: {auc:.4f}")
    
    avg_auc = total_auc / len(all_results)
    print(f"Average AUC: {avg_auc:.4f}")
    
    final_results_path = Path(config.SAVE_DIR) / 'final_results.json'
    with open(final_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()