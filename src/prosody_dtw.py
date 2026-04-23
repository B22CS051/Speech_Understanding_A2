"""
Prosody Extraction and Dynamic Time Warping (DTW) Module
Transfers prosodic features (F0, energy) between sources
"""

import numpy as np
import librosa
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import Tuple, List, Dict


class ProsodyExtractor:
    """Extract prosodic features (F0, energy, duration)"""
    
    def __init__(self, sr=16000, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
        
    def extract_f0(self, audio: np.ndarray, f0_min=50, f0_max=500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency using librosa's pyin
        
        Returns:
            f0: array of F0 values
            voiced_flag: array indicating voiced/unvoiced frames
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=f0_min,
            fmax=f0_max,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Interpolate NaN values
        valid_idx = ~np.isnan(f0)
        if np.sum(valid_idx) > 1:
            f = interp1d(
                np.where(valid_idx)[0],
                f0[valid_idx],
                kind='linear',
                fill_value='extrapolate'
            )
            f0 = f(np.arange(len(f0)))
            f0 = np.clip(f0, f0_min, f0_max)
        else:
            f0[np.isnan(f0)] = 0
        
        return f0, voiced_flag
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract frame-level energy"""
        S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
        energy = np.sqrt(np.mean(S**2, axis=0))
        # Log scale
        energy_db = librosa.power_to_db(energy**2 + 1e-9, ref=np.max)
        return energy_db
    
    def extract_duration(self, audio: np.ndarray) -> float:
        """Get total duration in seconds"""
        return len(audio) / self.sr
    
    def extract_mfcc_delta(self, audio: np.ndarray, n_mfcc=13) -> np.ndarray:
        """Extract MFCC derivatives (delta features)"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc, 
                                    hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        
        return np.vstack([mfcc, mfcc_delta, mfcc_delta_delta])
    
    def extract_all(self, audio: np.ndarray) -> Dict:
        """Extract all prosodic features"""
        f0, voiced_flag = self.extract_f0(audio)
        energy = self.extract_energy(audio)
        duration = self.extract_duration(audio)
        mfcc_features = self.extract_mfcc_delta(audio)
        
        return {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'energy': energy,
            'duration': duration,
            'mfcc_features': mfcc_features,
            'hop_length': self.hop_length,
            'sr': self.sr,
            'num_frames': len(f0),
            'timestamps': np.arange(len(f0)) * self.hop_length / self.sr
        }


class DynamicTimeWarping:
    """Dynamic Time Warping for prosody mapping"""
    
    @staticmethod
    def compute_dtw(source_features: np.ndarray, 
                   target_features: np.ndarray) -> Tuple[float, List]:
        """
        Compute DTW distance and warping path
        
        Args:
            source_features: (n_frames, n_features)
            target_features: (m_frames, n_features)
            
        Returns:
            distance: DTW distance
            path: warping path as list of (i, j) tuples
        """
        # Ensure 2D
        if source_features.ndim == 1:
            source_features = source_features.reshape(-1, 1)
        if target_features.ndim == 1:
            target_features = target_features.reshape(-1, 1)
        
        # Compute DTW
        distance, path = fastdtw(source_features, target_features, 
                                 dist=euclidean)
        
        return distance, path
    
    @staticmethod
    def warp_feature(source_feature: np.ndarray, 
                    path: List[Tuple[int, int]]) -> np.ndarray:
        """
        Warp feature using DTW path
        
        Args:
            source_feature: (n_frames,) array
            path: DTW warping path
            
        Returns:
            warped_feature: (m_frames,) array (same length as target)
        """
        # Extract unique target indices from path
        target_indices = np.array([j for i, j in path])
        unique_indices = np.unique(target_indices)
        
        # For each unique target index, take mean of corresponding source values
        warped = np.zeros(unique_indices.max() + 1)
        
        for src_idx, tgt_idx in path:
            if tgt_idx < len(warped):
                warped[tgt_idx] = source_feature[src_idx]
        
        return warped


class ProsodyTransfer:
    """Transfer prosodic features from source to target"""
    
    def __init__(self, sr=16000, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length
        self.extractor = ProsodyExtractor(sr, hop_length)
        self.dtw = DynamicTimeWarping()
        
    def transfer_f0_contour(self, source_audio: np.ndarray, 
                           target_audio: np.ndarray, 
                           alpha: float = 1.0) -> np.ndarray:
        """
        Transfer F0 contour from source to target audio
        
        Args:
            source_audio: source speech audio
            target_audio: target speech audio
            alpha: scaling factor for F0 transfer strength
            
        Returns:
            modified_f0: F0 contour to apply to target
        """
        # Extract F0
        source_f0, _ = self.extractor.extract_f0(source_audio)
        target_f0, _ = self.extractor.extract_f0(target_audio)
        
        # Normalize F0 values (remove DC offset)
        source_f0_norm = source_f0 - np.mean(source_f0[source_f0 > 0])
        target_f0_norm = target_f0 - np.mean(target_f0[target_f0 > 0])
        target_f0_baseline = np.mean(target_f0[target_f0 > 0])
        
        # Compute DTW
        _, path = self.dtw.compute_dtw(
            source_f0_norm.reshape(-1, 1),
            target_f0_norm.reshape(-1, 1)
        )
        
        # Warp
        warped_f0_norm = self.dtw.warp_feature(source_f0_norm, path)
        
        # Ensure same length as target
        if len(warped_f0_norm) != len(target_f0):
            # Interpolate to match length
            interp_f = interp1d(
                np.arange(len(warped_f0_norm)),
                warped_f0_norm,
                kind='linear',
                fill_value='extrapolate'
            )
            warped_f0_norm = interp_f(
                np.linspace(0, len(warped_f0_norm)-1, len(target_f0))
            )
        
        # Apply to target baseline
        modified_f0 = target_f0_baseline + alpha * warped_f0_norm
        modified_f0 = np.clip(modified_f0, 50, 500)
        
        return modified_f0
    
    def transfer_energy_contour(self, source_audio: np.ndarray,
                               target_audio: np.ndarray,
                               alpha: float = 1.0) -> np.ndarray:
        """Transfer energy contour from source to target"""
        source_energy = self.extractor.extract_energy(source_audio)
        target_energy = self.extractor.extract_energy(target_audio)
        
        # Normalize
        source_energy_norm = (source_energy - np.mean(source_energy)) / (np.std(source_energy) + 1e-9)
        target_energy_norm = (target_energy - np.mean(target_energy)) / (np.std(target_energy) + 1e-9)
        target_energy_baseline = np.mean(target_energy)
        
        # DTW
        _, path = self.dtw.compute_dtw(
            source_energy_norm.reshape(-1, 1),
            target_energy_norm.reshape(-1, 1)
        )
        
        # Warp
        warped_energy_norm = self.dtw.warp_feature(source_energy_norm, path)
        
        # Match length
        if len(warped_energy_norm) != len(target_energy):
            interp_e = interp1d(
                np.arange(len(warped_energy_norm)),
                warped_energy_norm,
                kind='linear',
                fill_value='extrapolate'
            )
            warped_energy_norm = interp_e(
                np.linspace(0, len(warped_energy_norm)-1, len(target_energy))
            )
        
        modified_energy = target_energy_baseline + alpha * warped_energy_norm
        return modified_energy
    
    def transfer_all_prosody(self, source_audio: np.ndarray,
                            target_audio: np.ndarray,
                            f0_alpha: float = 0.8,
                            energy_alpha: float = 0.6) -> Dict:
        """
        Transfer all prosodic features
        
        Returns:
            dict with modified prosodic features
        """
        modified_f0 = self.transfer_f0_contour(source_audio, target_audio, f0_alpha)
        modified_energy = self.transfer_energy_contour(source_audio, target_audio, energy_alpha)
        
        return {
            'f0': modified_f0,
            'energy': modified_energy,
            'f0_alpha': f0_alpha,
            'energy_alpha': energy_alpha
        }


class ProsodyModifier:
    """Apply prosodic modifications to audio (requires external tools)"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def apply_f0_modification(self, audio: np.ndarray, 
                             target_f0: np.ndarray) -> np.ndarray:
        """
        Apply F0 modification to audio
        Note: This is a simplified version. Production use requires
        tools like psola or WORLD vocoder
        """
        # For now, return original audio
        # In production, would use librosa's piptrack + phase vocoder
        return audio
    
    def apply_energy_modification(self, audio: np.ndarray,
                                 target_energy: np.ndarray,
                                 hop_length: int = 512) -> np.ndarray:
        """Apply energy modification to audio"""
        # Extract current energy
        S = np.abs(librosa.stft(audio, hop_length=hop_length))
        current_energy = np.sqrt(np.mean(S**2, axis=0))
        
        # Compute scaling factors
        scaling = target_energy / (current_energy + 1e-9)
        scaling = np.clip(scaling, 0.5, 2.0)  # Limit extreme scaling
        
        # Apply frame-wise scaling (simplified)
        # In production, use proper vocoder
        modified = audio.copy()
        for i, scale in enumerate(scaling):
            start = i * hop_length
            end = start + hop_length
            if end <= len(modified):
                modified[start:end] *= scale
        
        return modified


# Type hints
from typing import Dict

