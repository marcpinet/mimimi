import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix, Specificity
import pickle

from model import ComplexAnomalyDetector
from dataset import MIMIIDataset
from config import Config
from utils import find_youden_threshold


class Evaluator:
    def __init__(self, machine_type, model_path, config=None):
        self.config = config or Config()
        self.machine_type = machine_type
        self.device = torch.device(self.config.DEVICE)
        
        self.model = None
        self.test_loader = None
        self.model_path = model_path
        
        self.setup_data()
        self.load_model()
        self.setup_metrics()
        
    def setup_metrics(self):
        self.classification_metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device),
            'precision': Precision(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'recall': Recall(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'f1': F1Score(task='multiclass', num_classes=self.num_classes, average='weighted').to(self.device),
            'precision_per_class': Precision(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
            'recall_per_class': Recall(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
            'f1_per_class': F1Score(task='multiclass', num_classes=self.num_classes, average=None).to(self.device),
            'confusion_matrix': ConfusionMatrix(task='multiclass', num_classes=self.num_classes).to(self.device)
        }
        
        self.anomaly_metrics = {
            'auroc': AUROC(task='binary').to(self.device),
            'precision': Precision(task='binary').to(self.device),
            'recall': Recall(task='binary').to(self.device),
            'f1': F1Score(task='binary').to(self.device),
            'specificity': Specificity(task='binary').to(self.device),
            'confusion_matrix': ConfusionMatrix(task='binary').to(self.device)
        }
        
    def reset_metrics(self):
        for metric in self.classification_metrics.values():
            metric.reset()
        for metric in self.anomaly_metrics.values():
            metric.reset()
        
    def setup_data(self):
        test_dataset = MIMIIDataset(
            self.config.DATA_PATH,
            self.machine_type,
            split='test',
            sr=self.config.AUDIO_SR,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_frames=self.config.N_FRAMES,
            augment=False,
            normalize= self.config.NORMALIZE
        )
        
        def test_collate_fn(batch):
            data, machine_id, label = zip(*batch)
            
            real_parts = []
            imag_parts = []
            
            for d in data:
                real_part, imag_part = d
                real_parts.append(real_part)
                imag_parts.append(imag_part)
            
            real_batch = torch.stack(real_parts, 0)
            imag_batch = torch.stack(imag_parts, 0)
            machine_id_batch = torch.stack(machine_id, 0)
            label_batch = torch.stack(label, 0)
            
            return (real_batch, imag_batch), machine_id_batch, label_batch
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=test_collate_fn
        )
        self.num_classes = test_dataset.num_machine_ids
        
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = ComplexAnomalyDetector(num_classes=checkpoint['num_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate_comprehensive(self):
        all_scores = []
        all_labels = []
        all_machine_ids = []
        all_predictions = []
        all_logits = []
        
        self.reset_metrics()
        
        with torch.no_grad():
            for data, machine_id, label in tqdm(self.test_loader, desc='Evaluation'):
                real_batch, imag_batch = data
                real_batch = real_batch.to(self.device)
                imag_batch = imag_batch.to(self.device)
                machine_id = machine_id.to(self.device).squeeze()
                label = label.squeeze()
                
                anomaly_scores = self.model.get_anomaly_score((real_batch, imag_batch), machine_id)
                
                logits_magnitude, logits_complex, logits_total = self.model((real_batch, imag_batch))
                predictions = torch.argmax(logits_total, dim=1)
                
                self.classification_metrics['accuracy'].update(predictions, machine_id)
                self.classification_metrics['precision'].update(predictions, machine_id)
                self.classification_metrics['recall'].update(predictions, machine_id)
                self.classification_metrics['f1'].update(predictions, machine_id)
                self.classification_metrics['precision_per_class'].update(predictions, machine_id)
                self.classification_metrics['recall_per_class'].update(predictions, machine_id)
                self.classification_metrics['f1_per_class'].update(predictions, machine_id)
                self.classification_metrics['confusion_matrix'].update(predictions, machine_id)
                
                self.anomaly_metrics['auroc'].update(anomaly_scores, label.to(self.device))
                
                all_scores.extend(anomaly_scores.cpu().numpy())
                all_labels.extend(label.numpy())
                all_machine_ids.extend(machine_id.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.extend(logits_total.cpu().numpy())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_machine_ids = np.array(all_machine_ids)
        all_predictions = np.array(all_predictions)
        all_logits = np.array(all_logits)
        
        optimal_threshold = find_youden_threshold(all_scores, all_labels)
        binary_predictions = (all_scores > optimal_threshold).astype(int)
        
        binary_predictions_tensor = torch.tensor(binary_predictions).to(self.device)
        labels_tensor = torch.tensor(all_labels).to(self.device)
        
        self.anomaly_metrics['precision'].update(binary_predictions_tensor, labels_tensor)
        self.anomaly_metrics['recall'].update(binary_predictions_tensor, labels_tensor)
        self.anomaly_metrics['f1'].update(binary_predictions_tensor, labels_tensor)
        self.anomaly_metrics['specificity'].update(binary_predictions_tensor, labels_tensor)
        self.anomaly_metrics['confusion_matrix'].update(binary_predictions_tensor, labels_tensor)
        
        classification_accuracy = self.classification_metrics['accuracy'].compute().item()
        classification_precision = self.classification_metrics['precision'].compute().item()
        classification_recall = self.classification_metrics['recall'].compute().item()
        classification_f1 = self.classification_metrics['f1'].compute().item()
        classification_precision_per_class = self.classification_metrics['precision_per_class'].compute().cpu().numpy()
        classification_recall_per_class = self.classification_metrics['recall_per_class'].compute().cpu().numpy()
        classification_f1_per_class = self.classification_metrics['f1_per_class'].compute().cpu().numpy()
        classification_confusion_matrix = self.classification_metrics['confusion_matrix'].compute().cpu().numpy()
        
        anomaly_auroc = self.anomaly_metrics['auroc'].compute().item()
        anomaly_precision = self.anomaly_metrics['precision'].compute().item()
        anomaly_recall = self.anomaly_metrics['recall'].compute().item()
        anomaly_f1 = self.anomaly_metrics['f1'].compute().item()
        anomaly_specificity = self.anomaly_metrics['specificity'].compute().item()
        anomaly_confusion_matrix = self.anomaly_metrics['confusion_matrix'].compute().cpu().numpy()
        
        overall_auc = roc_auc_score(all_labels, all_scores)
        
        machine_id_results = {}
        for mid in np.unique(all_machine_ids):
            mask = all_machine_ids == mid
            if len(np.unique(all_labels[mask])) > 1:
                auc = roc_auc_score(all_labels[mask], all_scores[mask])
                machine_id_results[int(mid)] = {
                    'auc': auc,
                    'num_normal': np.sum(all_labels[mask] == 0),
                    'num_anomaly': np.sum(all_labels[mask] == 1)
                }
        
        results = {
            'machine_type': self.machine_type,
            'overall_auc': overall_auc,
            'machine_id_results': machine_id_results,
            'classification_metrics': {
                'accuracy': classification_accuracy,
                'precision': classification_precision,
                'recall': classification_recall,
                'f1': classification_f1,
                'precision_per_class': classification_precision_per_class.tolist(),
                'recall_per_class': classification_recall_per_class.tolist(),
                'f1_per_class': classification_f1_per_class.tolist(),
                'confusion_matrix': classification_confusion_matrix.tolist()
            },
            'anomaly_detection_metrics': {
                'auroc': anomaly_auroc,
                'precision': anomaly_precision,
                'recall': anomaly_recall,
                'f1': anomaly_f1,
                'specificity': anomaly_specificity,
                'confusion_matrix': anomaly_confusion_matrix.tolist(),
                'threshold': optimal_threshold
            },
            'num_test_samples': len(all_scores),
            'num_normal': np.sum(all_labels == 0),
            'num_anomaly': np.sum(all_labels == 1)
        }
        
        return results, all_scores, all_labels, all_machine_ids, all_predictions, all_logits
    
    def plot_results(self, results, scores, labels, machine_ids, predictions, save_path=None):
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[0, 0].plot(fpr, tpr, label=f'AUC = {results["overall_auc"]:.3f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title(f'ROC Curve - {self.machine_type}')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        axes[0, 1].hist(normal_scores, bins=30, alpha=0.7, label='Normal', density=True)
        axes[0, 1].hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', density=True)
        axes[0, 1].axvline(results['anomaly_detection_metrics']['threshold'], color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Anomaly Score')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        machine_aucs = [results['machine_id_results'][mid]['auc'] for mid in sorted(results['machine_id_results'].keys())]
        machine_ids_sorted = sorted(results['machine_id_results'].keys())
        
        axes[0, 2].bar(range(len(machine_aucs)), machine_aucs)
        axes[0, 2].set_xlabel('Machine ID')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].set_title('AUC by Machine ID')
        axes[0, 2].set_xticks(range(len(machine_ids_sorted)))
        axes[0, 2].set_xticklabels([f'ID {mid}' for mid in machine_ids_sorted])
        
        score_by_machine = {}
        for mid in np.unique(machine_ids):
            mask = machine_ids == mid
            score_by_machine[f'ID {int(mid)}'] = scores[mask]
        
        axes[1, 0].boxplot(score_by_machine.values(), labels=score_by_machine.keys())
        axes[1, 0].set_xlabel('Machine ID')
        axes[1, 0].set_ylabel('Anomaly Score')
        axes[1, 0].set_title('Score Distribution by Machine ID')
        
        classification_cm = np.array(results['classification_metrics']['confusion_matrix'])
        im1 = axes[1, 1].imshow(classification_cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Classification Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('True')
        
        for i in range(classification_cm.shape[0]):
            for j in range(classification_cm.shape[1]):
                axes[1, 1].text(j, i, format(classification_cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if classification_cm[i, j] > classification_cm.max() / 2. else "black")
        
        anomaly_cm = np.array(results['anomaly_detection_metrics']['confusion_matrix'])
        im2 = axes[1, 2].imshow(anomaly_cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 2].set_title('Anomaly Detection Confusion Matrix')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('True')
        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_yticks([0, 1])
        axes[1, 2].set_xticklabels(['Normal', 'Anomaly'])
        axes[1, 2].set_yticklabels(['Normal', 'Anomaly'])
        
        for i in range(anomaly_cm.shape[0]):
            for j in range(anomaly_cm.shape[1]):
                axes[1, 2].text(j, i, format(anomaly_cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if anomaly_cm[i, j] > anomaly_cm.max() / 2. else "black")
        
        classification_metrics_values = [
            results['classification_metrics']['accuracy'],
            results['classification_metrics']['precision'],
            results['classification_metrics']['recall'],
            results['classification_metrics']['f1']
        ]
        
        axes[2, 0].bar(['Accuracy', 'Precision', 'Recall', 'F1'], classification_metrics_values)
        axes[2, 0].set_title('Classification Metrics')
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].set_ylim(0, 1)
        
        anomaly_metrics_values = [
            results['anomaly_detection_metrics']['auroc'],
            results['anomaly_detection_metrics']['precision'],
            results['anomaly_detection_metrics']['recall'],
            results['anomaly_detection_metrics']['f1'],
            results['anomaly_detection_metrics']['specificity']
        ]
        
        axes[2, 1].bar(['AUROC', 'Precision', 'Recall', 'F1', 'Specificity'], anomaly_metrics_values)
        axes[2, 1].set_title('Anomaly Detection Metrics')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].set_ylim(0, 1)
        
        per_class_f1 = results['classification_metrics']['f1_per_class']
        axes[2, 2].bar(range(len(per_class_f1)), per_class_f1)
        axes[2, 2].set_title('F1 Score per Machine ID')
        axes[2, 2].set_xlabel('Machine ID')
        axes[2, 2].set_ylabel('F1 Score')
        axes[2, 2].set_xticks(range(len(per_class_f1)))
        axes[2, 2].set_xticklabels([f'ID {i}' for i in range(len(per_class_f1))])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_with_threshold(self, scores, labels, threshold):
        predictions = (scores > threshold).astype(int)
        
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tp = np.sum((predictions == 1) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def evaluate_model(machine_type, model_path, config=None):
    evaluator = Evaluator(machine_type, model_path, config)
    
    print(f"Evaluating {machine_type} model...")
    results, scores, labels, machine_ids, predictions, logits = evaluator.evaluate_comprehensive()
    
    print(f"\nResults for {machine_type}:")
    print(f"Overall AUC: {results['overall_auc']:.4f}")
    print(f"Classification Accuracy: {results['classification_metrics']['accuracy']:.4f}")
    print(f"Classification F1: {results['classification_metrics']['f1']:.4f}")
    print(f"Anomaly Detection AUROC: {results['anomaly_detection_metrics']['auroc']:.4f}")
    print(f"Anomaly Detection F1: {results['anomaly_detection_metrics']['f1']:.4f}")
    print(f"Test samples: {results['num_test_samples']} ({results['num_normal']} normal, {results['num_anomaly']} anomaly)")
    
    print("\nMachine ID Results:")
    for mid, mid_results in results['machine_id_results'].items():
        print(f"  ID {mid}: AUC = {mid_results['auc']:.4f} ({mid_results['num_normal']} normal, {mid_results['num_anomaly']} anomaly)")
    
    optimal_threshold = find_youden_threshold(scores, labels)
    threshold_results = evaluator.evaluate_with_threshold(scores, labels, optimal_threshold)
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Precision: {threshold_results['precision']:.4f}")
    print(f"Recall: {threshold_results['recall']:.4f}")
    print(f"F1-Score: {threshold_results['f1']:.4f}")
    print(f"Accuracy: {threshold_results['accuracy']:.4f}")
    print(f"Specificity: {threshold_results['specificity']:.4f}")
    
    save_path = Path(config.SAVE_DIR) / f'{machine_type}_evaluation.png'
    evaluator.plot_results(results, scores, labels, machine_ids, predictions, save_path)

    results_json_safe = convert_numpy_types(results)
    
    metrics_save_path = Path(config.SAVE_DIR) / f'{machine_type}_detailed_metrics.pkl'
    detailed_metrics = {
        'results': results_json_safe,
        'scores': scores,
        'labels': labels,
        'machine_ids': machine_ids,
        'predictions': predictions,
        'logits': logits,
        'threshold_results': convert_numpy_types(threshold_results)
    }
    
    with open(metrics_save_path, 'wb') as f:
        pickle.dump(detailed_metrics, f)
    
    return results_json_safe


def main(config=None):
    if config is None:
        config = Config()
    
    all_results = {}
    
    for machine_type in config.MACHINE_TYPES:
        model_path = Path(config.SAVE_DIR) / f'{machine_type}_best_model.pth'
        
        if model_path.exists():
            print(f"\n{'='*60}")
            print(f"Evaluating {machine_type} model")
            print(f"{'='*60}")
            
            results = evaluate_model(machine_type, model_path, config)
            all_results[machine_type] = results
        else:
            print(f"Model not found for {machine_type}: {model_path}")
    
    if all_results:
        print(f"\n{'='*60}")
        print("FINAL EVALUATION RESULTS")
        print(f"{'='*60}")
        
        total_auc = 0
        total_anomaly_auroc = 0
        total_anomaly_f1 = 0
        
        for machine_type, results in all_results.items():
            auc = results['overall_auc']
            anomaly_auroc = results['anomaly_detection_metrics']['auroc']
            anomaly_f1 = results['anomaly_detection_metrics']['f1']
            
            total_auc += auc
            total_anomaly_auroc += anomaly_auroc
            total_anomaly_f1 += anomaly_f1
            
            print(f"{machine_type}: AUC {auc:.4f}, AUROC {anomaly_auroc:.4f}, F1 {anomaly_f1:.4f}")
        
        avg_auc = total_auc / len(all_results)
        avg_anomaly_auroc = total_anomaly_auroc / len(all_results)
        avg_anomaly_f1 = total_anomaly_f1 / len(all_results)
        
        print(f"Average - AUC: {avg_auc:.4f}, AUROC: {avg_anomaly_auroc:.4f}, F1: {avg_anomaly_f1:.4f}")
        
        # https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0000586.pdf
        comparison_with_paper = {
            'fan': 89.55,
            'pump': 96.40,
            'slider': 98.75,
            'valve': 96.87,
            'average': 95.39
        }
        
        print(f"\nComparison with Paper Results:")
        for machine_type, results in all_results.items():
            our_auc = results['overall_auc'] * 100
            paper_auc = comparison_with_paper.get(machine_type, 0)
            diff = our_auc - paper_auc
            print(f"{machine_type}: Ours {our_auc:.2f}% vs Paper {paper_auc:.2f}% (diff: {diff:+.2f}%)")
        
        our_avg = avg_auc * 100
        paper_avg = comparison_with_paper['average']
        diff_avg = our_avg - paper_avg
        print(f"Average: Ours {our_avg:.2f}% vs Paper {paper_avg:.2f}% (diff: {diff_avg:+.2f}%)")
        
        evaluation_results_path = Path(config.SAVE_DIR) / 'evaluation_results.json'
        with open(evaluation_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()