"""
Anti-Spoofing / Countermeasure (CM) Module
Detects spoofed/synthetic audio vs real speech
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from scipy import signal
from typing import Tuple, Dict


class LFCCFeatureExtractor:
    """Extract LFCC (Linear Frequency Cepstral Coefficients) features"""
    
    def __init__(self, sr=16000, n_fft=2048, hop_length=512, n_coeffs=13):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_coeffs = n_coeffs
        
    def compute_lfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute LFCC features
        
        Returns:
            lfcc: (n_coeffs, n_frames)
        """
        # Compute linear spectrogram (not mel-scaled)
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, 
                               hop_length=self.hop_length))
        
        # DCT on linear spectrum
        lfcc = librosa.feature.mfcc(S=S, n_mfcc=self.n_coeffs)
        
        # Normalize
        lfcc = (lfcc - lfcc.mean(axis=1, keepdims=True)) / (lfcc.std(axis=1, keepdims=True) + 1e-9)
        
        return lfcc
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract full LFCC feature vector
        
        Returns:
            features: (n_frames, n_coeffs + deltas + delta-deltas)
        """
        lfcc = self.compute_lfcc(audio)
        
        # Compute dynamic features
        delta = librosa.feature.delta(lfcc)
        delta_delta = librosa.feature.delta(lfcc, order=2)
        
        # Combine
        features = np.vstack([lfcc, delta, delta_delta])  # (39, n_frames)
        
        return features.T  # (n_frames, 39)


class CQCCFeatureExtractor:
    """Extract CQCC (Constant-Q Cepstral Coefficients) features"""
    
    def __init__(self, sr=16000, hop_length=512, n_coeffs=13):
        self.sr = sr
        self.hop_length = hop_length
        self.n_coeffs = n_coeffs
        
    def compute_cqcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute CQCC features using constant-Q transform
        
        Returns:
            cqcc: (n_coeffs, n_frames)
        """
        # Compute constant-Q transform
        cqt = np.abs(librosa.cqt(audio, sr=self.sr, hop_length=self.hop_length))
        
        # Apply log scaling
        log_cqt = np.log(cqt + 1e-9)
        
        # Compute cepstrum via DCT
        cqcc = np.zeros((self.n_coeffs, cqt.shape[1]))
        for frame in range(cqt.shape[1]):
            cqcc[:, frame] = librosa.feature.delta(
                librosa.power_to_db(cqt[:, frame:frame+1] + 1e-9)
            ).flatten()[:self.n_coeffs]
        
        # Normalize
        cqcc = (cqcc - cqcc.mean(axis=1, keepdims=True)) / (cqcc.std(axis=1, keepdims=True) + 1e-9)
        
        return cqcc
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract full CQCC features with deltas"""
        cqcc = self.compute_cqcc(audio)
        delta = librosa.feature.delta(cqcc)
        delta_delta = librosa.feature.delta(cqcc, order=2)
        
        features = np.vstack([cqcc, delta, delta_delta])
        return features.T


class SpoofingClassifier(nn.Module):
    """CNN-based classifier for spoof detection"""
    
    def __init__(self, input_dim=39, hidden_dim=128):
        super().__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2)
        
        # Global average pooling effect
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Binary: Spoof or Bona Fide
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim, seq_len)
            
        Returns:
            logits: (batch_size, 2)
        """
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        
        # Classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class AntiSpoofingSystem:
    """End-to-end anti-spoofing system"""
    
    def __init__(self, feature_type='lfcc', device='cpu'):
        self.device = torch.device(device)
        self.feature_type = feature_type
        
        # Feature extractor
        if feature_type == 'lfcc':
            self.feature_extractor = LFCCFeatureExtractor()
            input_dim = 39
        else:  # cqcc
            self.feature_extractor = CQCCFeatureExtractor()
            input_dim = 39
        
        # Classifier
        self.classifier = SpoofingClassifier(input_dim=input_dim)
        self.classifier.to(self.device)
        self.classifier.eval()
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for classification"""
        return self.feature_extractor.extract_features(audio)
    
    def predict(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Predict if audio is spoofed or bona fide
        
        Returns:
            label: 'Bona Fide' or 'Spoof'
            confidence: confidence score (0-1)
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, dim, time)
        
        # Predict
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1)
        
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        label = 'Bona Fide' if pred_class == 0 else 'Spoof'
        
        return label, confidence
    
    def predict_batch(self, audio_list: list) -> list:
        """Batch prediction"""
        results = []
        for audio in audio_list:
            label, conf = self.predict(audio)
            results.append({'label': label, 'confidence': conf, 'audio': audio})
        return results


class SpoofDetectionEvaluator:
    """Evaluate anti-spoofing system performance"""
    
    @staticmethod
    def compute_eer(genuine_scores: np.ndarray,
                   spoof_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER)
        
        Args:
            genuine_scores: confidence scores for genuine (bona fide) samples
            spoof_scores: confidence scores for spoofed samples
            
        Returns:
            eer: Equal Error Rate
            threshold: threshold at EER
        """
        # Sort scores
        genuine_scores = np.sort(genuine_scores)
        spoof_scores = np.sort(spoof_scores)
        
        # Generate thresholds
        all_scores = np.concatenate([genuine_scores, spoof_scores])
        thresholds = np.linspace(0, 1, 1000)
        
        min_eer = 1.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            # False Positive Rate (spoof classified as genuine)
            fpr = np.mean(spoof_scores >= threshold)
            
            # False Negative Rate (genuine classified as spoof)
            fnr = np.mean(genuine_scores < threshold)
            
            eer = (fpr + fnr) / 2
            
            if eer < min_eer:
                min_eer = eer
                best_threshold = threshold
        
        return min_eer, best_threshold
    
    @staticmethod
    def compute_detection_cost_function(genuine_scores: np.ndarray,
                                       spoof_scores: np.ndarray,
                                       threshold: float,
                                       c_fa: float = 1.0,
                                       c_miss: float = 1.0,
                                       p_target: float = 0.05) -> float:
        """
        Compute detection cost function (DCF) for anti-spoofing
        
        Args:
            genuine_scores: confidence for bona fide
            spoof_scores: confidence for spoofed
            threshold: decision threshold
            c_fa: cost of false alarm
            c_miss: cost of miss
            p_target: target prior probability
            
        Returns:
            dcf: normalized detection cost
        """
        # False alarm rate
        fa = np.mean(spoof_scores >= threshold)
        
        # Miss rate
        miss = np.mean(genuine_scores < threshold)
        
        # Compute DCF
        dcf = c_miss * miss * p_target + c_fa * fa * (1 - p_target)
        
        # Normalize
        dcf_norm = dcf / min(c_miss * p_target, c_fa * (1 - p_target))
        
        return dcf_norm


class VoiceConversionDetection:
    """Detect voice conversion and manipulation artifacts"""
    
    @staticmethod
    def detect_artifacts(audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Detect common artifacts in voice conversion/synthesis
        
        Returns:
            dict with artifact indicators
        """
        # Check for prosody unnaturalness
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
        f0_variance = np.nanvar(f0[f0 > 0])
        
        # Check for spectral discontinuities
        spec = np.abs(librosa.stft(audio))
        spec_var = np.std(np.diff(spec, axis=1))
        
        # Check for phase coherence
        phase = np.angle(librosa.stft(audio))
        phase_coherence = np.abs(np.mean(np.exp(1j * phase), axis=0)).mean()
        
        return {
            'f0_variance': float(f0_variance),
            'spectral_variance': float(spec_var),
            'phase_coherence': float(phase_coherence),
            'likely_synthetic': phase_coherence < 0.3 or f0_variance < 10
        }
