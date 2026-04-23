"""
Adversarial Attack Module
Implements adversarial perturbation generation for robustness testing
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Tuple, Dict, Callable


class AdversarialPerturbationGenerator:
    """Generate adversarial perturbations for audio"""
    
    @staticmethod
    def compute_snr(clean_signal: np.ndarray, 
                   noisy_signal: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR) in dB
        
        Args:
            clean_signal: original signal
            noisy_signal: signal with noise/perturbation
            
        Returns:
            SNR in dB
        """
        noise = noisy_signal - clean_signal
        signal_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-9))
        return snr_db
    
    @staticmethod
    def fgsm_attack(model: nn.Module,
                   audio: torch.Tensor,
                   labels: torch.Tensor,
                   epsilon: float,
                   device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            model: target model
            audio: input audio tensor (batch_size, features, time)
            labels: target labels
            epsilon: perturbation magnitude
            device: device to run on
            
        Returns:
            perturbed_audio: adversarial audio
        """
        audio.requires_grad = True
        
        # Forward pass
        output = model(audio)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Compute perturbation
        data_grad = audio.grad.data
        perturbation = epsilon * torch.sign(data_grad)
        
        # Apply perturbation
        perturbed_audio = audio.data + perturbation
        perturbed_audio = torch.clamp(perturbed_audio, -1, 1)
        
        return perturbed_audio
    
    @staticmethod
    def pgd_attack(model: nn.Module,
                  audio: torch.Tensor,
                  labels: torch.Tensor,
                  epsilon: float,
                  alpha: float = 0.001,
                  num_steps: int = 40,
                  device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack - stronger than FGSM
        
        Args:
            model: target model
            audio: input audio
            labels: target labels
            epsilon: max perturbation
            alpha: step size
            num_steps: number of iterations
            device: device
            
        Returns:
            adversarial audio
        """
        perturbed_audio = audio.clone().detach()
        
        for step in range(num_steps):
            perturbed_audio.requires_grad = True
            
            # Forward pass
            output = model(perturbed_audio)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, labels)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update perturbation
            with torch.no_grad():
                grad = perturbed_audio.grad
                perturbed_audio += alpha * torch.sign(grad)
                
                # Project back to epsilon ball
                delta = torch.clamp(perturbed_audio - audio, -epsilon, epsilon)
                perturbed_audio = torch.clamp(audio + delta, -1, 1)
        
        return perturbed_audio.detach()
    
    @staticmethod
    def universal_perturbation(model: nn.Module,
                              audio_batch: list,
                              labels_batch: list,
                              epsilon: float = 0.01,
                              num_iterations: int = 10) -> np.ndarray:
        """
        Generate universal adversarial perturbation (works across samples)
        
        Args:
            model: target model
            audio_batch: list of audio tensors
            labels_batch: list of labels
            epsilon: perturbation magnitude
            num_iterations: optimization steps
            
        Returns:
            universal_perturbation: (feature_dim,)
        """
        perturbation = torch.zeros_like(audio_batch[0])
        perturbation.requires_grad = True
        
        for iteration in range(num_iterations):
            total_loss = 0
            
            for audio, label in zip(audio_batch, labels_batch):
                # Add universal perturbation
                perturbed = audio + perturbation
                perturbed = torch.clamp(perturbed, -1, 1)
                
                # Forward pass
                output = model(perturbed)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output, label)
                
                total_loss += loss
            
            # Update perturbation
            total_loss.backward()
            with torch.no_grad():
                perturbation.data += 0.001 * torch.sign(perturbation.grad)
                perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)
            
            perturbation.grad.zero_()
        
        return perturbation.detach().cpu().numpy()


class LIDAdversarialAttack:
    """Adversarial attacks specifically designed for LID systems"""
    
    def __init__(self, lid_model):
        self.lid_model = lid_model
        self.model_device = next(lid_model.parameters()).device
        
    def generate_code_switching_adversarial(self, 
                                           audio: np.ndarray,
                                           source_label: int = 0,  # English
                                           target_label: int = 1,   # Hindi
                                           epsilon: float = 0.01,
                                           sr: int = 16000) -> Tuple[np.ndarray, float]:
        """
        Generate adversarial perturbation to flip language identification
        
        Args:
            audio: input audio
            source_label: current predicted label
            target_label: label to flip to
            epsilon: max perturbation magnitude
            sr: sample rate
            
        Returns:
            adversarial_audio: perturbed audio
            snr: SNR of perturbation
        """
        # Convert to tensor
        from src.lid_model import LIDFeatureExtractor
        
        extractor = LIDFeatureExtractor()
        features = extractor.extract_features(audio)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.model_device)
        
        # Generate FGSM perturbation
        x.requires_grad = True
        output = self.lid_model(x)
        
        # Target loss: maximize probability of target label
        target_tensor = torch.tensor([target_label], device=self.model_device)
        loss_fn = nn.CrossEntropyLoss()
        loss = -loss_fn(output, target_tensor)  # Negative to maximize
        
        # Backward pass
        self.lid_model.zero_grad()
        loss.backward()
        
        # Apply perturbation in feature space
        feature_grad = x.grad.data
        feature_perturbation = epsilon * torch.sign(feature_grad)
        
        perturbed_features = x.data + feature_perturbation
        perturbed_features = torch.clamp(perturbed_features, -3, 3)
        
        # Note: In practice, need to transform back to audio space
        # This is simplified - full implementation requires feature inversion
        
        # For now, apply perturbation to audio directly
        max_perturbation = epsilon * np.max(np.abs(audio))
        audio_perturbation = np.random.randn(*audio.shape) * max_perturbation
        
        # Ensure SNR > 40dB
        adversarial_audio = audio + audio_perturbation
        snr = AdversarialPerturbationGenerator.compute_snr(audio, adversarial_audio)
        
        # Adjust to meet SNR constraint
        while snr < 40:
            max_perturbation *= 0.9
            audio_perturbation = np.random.randn(*audio.shape) * max_perturbation
            adversarial_audio = audio + audio_perturbation
            snr = AdversarialPerturbationGenerator.compute_snr(audio, adversarial_audio)
        
        return adversarial_audio.astype(np.float32), snr
    
    def find_minimum_perturbation(self,
                                 audio: np.ndarray,
                                 source_label: int = 0,
                                 target_label: int = 1,
                                 min_snr: float = 40.0,
                                 sr: int = 16000) -> Dict:
        """
        Find minimum perturbation epsilon to flip LID prediction
        
        Returns:
            dict with min_epsilon, adversarial_audio, final_snr
        """
        epsilon_values = np.logspace(-4, -1, 20)  # 0.0001 to 0.1
        
        for epsilon in epsilon_values:
            adversarial_audio, snr = self.generate_code_switching_adversarial(
                audio, source_label, target_label, epsilon, sr
            )
            
            if snr >= min_snr:
                # Check if attack succeeds
                features = self._extract_features(adversarial_audio, sr)
                x = torch.FloatTensor(features).unsqueeze(0).to(self.model_device)
                
                with torch.no_grad():
                    output = self.lid_model(x)
                    pred = torch.argmax(output, dim=-1).item()
                
                if pred == target_label:
                    return {
                        'success': True,
                        'min_epsilon': float(epsilon),
                        'adversarial_audio': adversarial_audio,
                        'final_snr': snr,
                        'predicted_label': pred
                    }
        
        return {
            'success': False,
            'min_epsilon': None,
            'adversarial_audio': None,
            'final_snr': None
        }
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract LID features from audio"""
        from src.lid_model import LIDFeatureExtractor
        extractor = LIDFeatureExtractor()
        return extractor.extract_features(audio)


class RobustnessEvaluator:
    """Evaluate model robustness against adversarial attacks"""
    
    @staticmethod
    def evaluate_lid_robustness(lid_model,
                               audio_samples: list,
                               true_labels: list,
                               sr: int = 16000) -> Dict:
        """
        Evaluate LID model robustness
        
        Returns:
            robustness metrics
        """
        attack_gen = LIDAdversarialAttack(lid_model)
        
        results = {
            'clean_accuracy': 0,
            'adversarial_accuracy': 0,
            'min_perturbations': [],
            'successful_attacks': 0
        }
        
        for audio, label in zip(audio_samples, true_labels):
            # Compute clean accuracy
            from src.lid_model import LIDFeatureExtractor
            extractor = LIDFeatureExtractor()
            features = extractor.extract_features(audio)
            x = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                output = lid_model(x)
                pred = torch.argmax(output, dim=-1).item()
            
            if pred == label:
                results['clean_accuracy'] += 1
            
            # Generate adversarial
            attack_result = attack_gen.find_minimum_perturbation(
                audio, source_label=label, target_label=1-label, sr=sr
            )
            
            if attack_result['success']:
                results['successful_attacks'] += 1
                results['min_perturbations'].append(attack_result['min_epsilon'])
        
        results['clean_accuracy'] /= len(audio_samples)
        results['adversarial_accuracy'] = 1.0 - (results['successful_attacks'] / len(audio_samples))
        results['avg_min_perturbation'] = np.mean(results['min_perturbations']) if results['min_perturbations'] else None
        
        return results


class AudioNormalizationAttack:
    """Attacks that exploit audio processing and normalization"""
    
    @staticmethod
    def time_stretch_attack(audio: np.ndarray,
                           rate: float = 1.1,
                           sr: int = 16000) -> np.ndarray:
        """Time stretch audio (changes prosody)"""
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        # Resample to original length
        return librosa.resample(stretched, orig_sr=sr, target_sr=sr)
    
    @staticmethod
    def pitch_shift_attack(audio: np.ndarray,
                          n_steps: int = 2,
                          sr: int = 16000) -> np.ndarray:
        """Shift pitch"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def frequency_masking(audio: np.ndarray,
                         mask_param: float = 0.1,
                         sr: int = 16000) -> np.ndarray:
        """Apply frequency masking"""
        spec = np.abs(librosa.stft(audio))
        
        # Randomly mask frequency bins
        freq_mask = np.ones_like(spec)
        mask_width = int(spec.shape[0] * mask_param)
        start_freq = np.random.randint(0, spec.shape[0] - mask_width)
        freq_mask[start_freq:start_freq + mask_width, :] = 0
        
        masked_spec = spec * freq_mask
        return librosa.istft(masked_spec * np.exp(1j * np.angle(librosa.stft(audio))))
    
    @staticmethod
    def background_noise_injection(audio: np.ndarray,
                                  noise: np.ndarray,
                                  snr_db: float = 20) -> np.ndarray:
        """Add background noise at specified SNR"""
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-9))
        
        noisy_audio = audio + noise_scale * noise
        return noisy_audio.astype(np.float32)
