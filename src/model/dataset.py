import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from pathlib import Path
import random
from glob import glob


class MIMIIDataset(Dataset):
    def __init__(self, data_path, machine_type, split='train', sr=16000, n_fft=1024, 
                 hop_length=512, n_frames=64, augment=True, normalize=False):
        self.data_path = Path(data_path)
        self.machine_type = machine_type
        self.split = split
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frames = n_frames
        self.augment = augment
        self.normalize = normalize

        self.machine_path = self.data_path / f'dev_data_{machine_type}' / machine_type / split

        self.files = []
        self.labels = []
        self.machine_ids = []

        self._load_file_paths()
        self._extract_machine_ids()

    def _load_file_paths(self):
        normal_files = list(self.machine_path.glob('normal_*.wav'))
        anomaly_files = list(self.machine_path.glob('anomaly_*.wav'))
        
        for file in normal_files:
            self.files.append(file)
            self.labels.append(0)
            
        for file in anomaly_files:
            self.files.append(file)
            self.labels.append(1)
            
    def _extract_machine_ids(self):
        machine_id_set = set()
        
        for file in self.files:
            filename = file.name
            if 'id_' in filename:
                machine_id = filename.split('id_')[1].split('_')[0]
                machine_id_set.add(machine_id)
        
        self.machine_id_to_idx = {mid: idx for idx, mid in enumerate(sorted(machine_id_set))}
        self.num_machine_ids = len(self.machine_id_to_idx)
        
        for file in self.files:
            filename = file.name
            if 'id_' in filename:
                machine_id = filename.split('id_')[1].split('_')[0]
                self.machine_ids.append(self.machine_id_to_idx[machine_id])
            else:
                self.machine_ids.append(0)
                
    def _load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        return y
    
    def _normalize_spectrum(self, real_part, imag_part):
        real_mean = np.mean(real_part)
        real_std = np.std(real_part) + 1e-8
        
        imag_mean = np.mean(imag_part)
        imag_std = np.std(imag_part) + 1e-8
        
        real_normalized = (real_part - real_mean) / real_std
        imag_normalized = (imag_part - imag_mean) / imag_std
        
        return real_normalized, imag_normalized
    
    def _compute_spectrum(self, y):
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        real_part = np.real(D)
        imag_part = np.imag(D)
        
        expected_freq_bins = self.n_fft // 2 + 1
        if real_part.shape[0] != expected_freq_bins:
            print(f"Warning: Expected {expected_freq_bins} frequency bins, got {real_part.shape[0]}")
            
        return real_part, imag_part
    
    def _extract_frames(self, spectrum_real, spectrum_imag):
        total_frames = spectrum_real.shape[1]
        
        if total_frames < self.n_frames:
            pad_width = self.n_frames - total_frames
            spectrum_real = np.pad(spectrum_real, ((0, 0), (0, pad_width)), mode='constant')
            spectrum_imag = np.pad(spectrum_imag, ((0, 0), (0, pad_width)), mode='constant')
        elif total_frames > self.n_frames:
            start_frame = random.randint(0, total_frames - self.n_frames)
            spectrum_real = spectrum_real[:, start_frame:start_frame + self.n_frames]
            spectrum_imag = spectrum_imag[:, start_frame:start_frame + self.n_frames]
            
        return spectrum_real, spectrum_imag
    
    def _augment_spectrum(self, spectrum_real, spectrum_imag):
        if not self.augment or self.split != 'train':
            return spectrum_real, spectrum_imag
        
        if random.random() < 0.2:
            signal_power = np.mean(spectrum_real**2 + spectrum_imag**2)
            noise_factor = random.uniform(0.01, 0.05) * np.sqrt(signal_power)
            
            noise_real = np.random.normal(0, noise_factor, spectrum_real.shape)
            noise_imag = np.random.normal(0, noise_factor, spectrum_imag.shape)
            spectrum_real = spectrum_real + noise_real
            spectrum_imag = spectrum_imag + noise_imag
        
        if random.random() < 0.15:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                mask_width = random.randint(2, 8)
                mask_start = random.randint(0, spectrum_real.shape[0] - mask_width)
                
                mask_factor = random.uniform(0.1, 0.3)
                spectrum_real[mask_start:mask_start+mask_width, :] *= mask_factor
                spectrum_imag[mask_start:mask_start+mask_width, :] *= mask_factor
        
        if random.random() < 0.1:
            shift = random.randint(-2, 2)
            if shift != 0:
                spectrum_real = np.roll(spectrum_real, shift, axis=0)
                spectrum_imag = np.roll(spectrum_imag, shift, axis=0)
        
        return spectrum_real, spectrum_imag
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        machine_id = self.machine_ids[idx]
        
        y = self._load_audio(file_path)
        spectrum_real, spectrum_imag = self._compute_spectrum(y)
        
        if self.normalize:
            spectrum_real, spectrum_imag = self._normalize_spectrum(spectrum_real, spectrum_imag)
        
        spectrum_real, spectrum_imag = self._extract_frames(spectrum_real, spectrum_imag)
        spectrum_real, spectrum_imag = self._augment_spectrum(spectrum_real, spectrum_imag)
        
        spectrum_real = torch.FloatTensor(spectrum_real).unsqueeze(0)
        spectrum_imag = torch.FloatTensor(spectrum_imag).unsqueeze(0)
        
        return (spectrum_real, spectrum_imag), torch.LongTensor([machine_id]), torch.LongTensor([label])
