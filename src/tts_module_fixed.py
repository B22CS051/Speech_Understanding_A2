"""
Text-to-Speech (TTS) Module
Implements voice cloning with speaker embeddings
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from scipy.signal import resample
from typing import Tuple, Dict


class SpeakerEmbeddingExtractor:
    """Extract speaker embedding from audio"""
    
    def __init__(self, sr=16000, embedding_dim=256):
        self.sr = sr
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        
    def _build_model(self) -> nn.Module:
        """Build speaker embedding model (simplified x-vector style)"""
        class SpeakerNet(nn.Module):
            def __init__(self, embedding_dim=256):
                super().__init__()
                self.fc1 = nn.Linear(832, 512)  # 64 MFCCs * 13 frames
                self.fc2 = nn.Linear(512, embedding_dim)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return SpeakerNet(embedding_dim=self.embedding_dim)
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=64, 
                                    n_fft=2048, hop_length=512)
        return mfcc
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: audio signal
            
        Returns:
            embedding: (embedding_dim,) speaker embedding
        """
        # Extract MFCC
        mfcc = self.extract_mfcc(audio)
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
        
        # Take center frames
        if mfcc.shape[1] > 13:
            start = (mfcc.shape[1] - 13) // 2
            mfcc = mfcc[:, start:start+13]
        else:
            # Pad if necessary
            pad_width = ((0, 0), (0, max(0, 13 - mfcc.shape[1])))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        
        # Flatten to vector
        mfcc_flat = mfcc.flatten()
        
        # Convert to tensor
        x = torch.FloatTensor(mfcc_flat).unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(x)
        
        return embedding.cpu().numpy()[0]


class TTSGenerator:
    """Generate speech from text with speaker embedding"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        
    def generate_from_embedding(self, text: str, 
                               speaker_embedding: np.ndarray,
                               duration: float = 5.0) -> np.ndarray:
        """
        Generate speech with speaker embedding
        
        Note: This is a simplified version. In production, would use
        a full TTS model like VITS, FastSpeech, or Meta MMS.
        
        Args:
            text: text to synthesize
            speaker_embedding: speaker embedding vector
            duration: approximate duration in seconds
            
        Returns:
            synthesized audio
        """
        # Generate placeholder audio with correct duration
        # In production, this would use a proper TTS model
        num_samples = int(duration * self.sr)
        audio = self._synthesize_placeholder(num_samples, speaker_embedding)
        
        return audio
    
    def _synthesize_placeholder(self, num_samples: int, 
                               embedding: np.ndarray) -> np.ndarray:
        """Generate placeholder audio using embedding characteristics"""
        # Use embedding to modulate generated signal
        embedding_mean = embedding.mean()
        embedding_std = embedding.std()
        
        # Generate base signal (frequency based on embedding)
        t = np.arange(num_samples) / self.sr
        fundamental_freq = 100 + 50 * embedding_mean
        base_signal = np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add harmonics for richness
        for harmonic in [2, 3, 4]:
            harmonic_signal = np.sin(2 * np.pi * fundamental_freq * harmonic * t)
            base_signal = base_signal + 0.3 * harmonic_signal / harmonic
        
        # Apply amplitude modulation
        envelope = np.sin(np.linspace(0, np.pi, num_samples))
        audio = base_signal * envelope
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        return audio.astype(np.float32)
    
    def synthesize_segment(self, text: str,
                          speaker_embedding: np.ndarray,
                          prosody_features: Dict = None) -> np.ndarray:
        """
        Synthesize with prosody features
        
        Args:
            text: input text
            speaker_embedding: speaker embedding
            prosody_features: dict with 'f0', 'energy'
            
        Returns:
            synthesized audio
        """
        # Estimate duration from text
        words = text.split()
        avg_word_duration = 0.5  # seconds per word
        estimated_duration = len(words) * avg_word_duration
        
        return self.generate_from_embedding(text, speaker_embedding, 
                                           estimated_duration)


class VoiceCloningPipeline:
    """End-to-end voice cloning pipeline"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        self.embedding_extractor = SpeakerEmbeddingExtractor(sr=16000)
        self.tts_generator = TTSGenerator(sr=sr)
        self.reference_embedding = None
        
    def set_reference_voice(self, audio_path: str):
        """
        Set reference voice for cloning
        
        Args:
            audio_path: path to reference audio (60 seconds)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract speaker embedding
        self.reference_embedding = self.embedding_extractor.extract_embedding(audio)
        
        return self.reference_embedding
    
    def generate_speech(self, text: str,
                       prosody_features: Dict = None) -> np.ndarray:
        """
        Generate speech using learned voice
        
        Args:
            text: text to synthesize
            prosody_features: prosody contours to apply
            
        Returns:
            synthesized audio
        """
        if self.reference_embedding is None:
            raise ValueError("Reference voice not set. Call set_reference_voice first.")
        
        # Generate speech
        audio = self.tts_generator.synthesize_segment(
            text,
            self.reference_embedding,
            prosody_features
        )
        
        # Resample if needed
        if self.tts_generator.sr != self.sr:
            num_samples = int(len(audio) * self.sr / self.tts_generator.sr)
            audio = resample(audio, num_samples)
        
        return audio
    
    def synthesize_full_lecture(self, transcript_segments: list,
                               prosody_per_segment: Dict = None,
                               combine_strategy: str = 'concatenate') -> np.ndarray:
        """
        Synthesize full lecture from segments
        
        Args:
            transcript_segments: list of text segments
            prosody_per_segment: dict mapping segment index to prosody features
            combine_strategy: how to combine segments ('concatenate', 'crossfade')
            
        Returns:
            full synthesized audio
        """
        generated_segments = []
        for i, text_segment in enumerate(transcript_segments):
            prosody = prosody_per_segment.get(i) if prosody_per_segment else None
            audio = self.generate_speech(text_segment, prosody)
            generated_segments.append(audio)
        
        if combine_strategy == 'concatenate':
            return np.concatenate(generated_segments)
        else:
            return self._combine_with_crossfade(generated_segments)
    
    def _combine_with_crossfade(self, audio_segments: list, 
                               crossfade_samples: int = 4410) -> np.ndarray:
        """Combine segments with crossfade"""
        if not audio_segments:
            return np.array([])
        
        result = audio_segments[0]
        for segment in audio_segments[1:]:
            # Simple crossfade
            crossfade_len = min(crossfade_samples, len(result), len(segment))
            
            fade_out = np.linspace(1, 0, crossfade_len)
            fade_in = np.linspace(0, 1, crossfade_len)
            
            # Mix overlapping region
            overlap = result[-crossfade_len:] * fade_out + segment[:crossfade_len] * fade_in
            
            # Concatenate
            result = np.concatenate([result[:-crossfade_len], overlap, segment[crossfade_len:]])
        
        return result


class SpeakerVerification:
    """Verify speaker characteristics in synthetic audio"""
    
    @staticmethod
    def verify_speaker_match(reference_audio: np.ndarray,
                            synthetic_audio: np.ndarray,
                            sr: int = 16000) -> float:
        """
        Compute speaker similarity between reference and synthetic
        
        Returns:
            similarity score (0-1)
        """
        extractor = SpeakerEmbeddingExtractor(sr=sr)
        
        ref_embedding = extractor.extract_embedding(reference_audio)
        synth_embedding = extractor.extract_embedding(synthetic_audio)
        
        # Cosine similarity
        cos_sim = np.dot(ref_embedding, synth_embedding) / \
                  (np.linalg.norm(ref_embedding) * np.linalg.norm(synth_embedding) + 1e-9)
        
        return (cos_sim + 1) / 2  # Normalize to [0, 1]
    
    @staticmethod
    def compute_similarity_metrics(reference_audio: np.ndarray,
                                  synthetic_audio: np.ndarray,
                                  sr: int = 16000) -> Dict:
        """Compute comprehensive similarity metrics"""
        extractor = SpeakerEmbeddingExtractor(sr=sr)
        
        ref_emb = extractor.extract_embedding(reference_audio)
        synth_emb = extractor.extract_embedding(synthetic_audio)
        
        # Cosine similarity
        cos_sim = np.dot(ref_emb, synth_emb) / \
                  (np.linalg.norm(ref_emb) * np.linalg.norm(synth_emb) + 1e-9)
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(ref_emb - synth_emb)
        
        return {
            'cosine_similarity': float(cos_sim),
            'euclidean_distance': float(euclidean_dist),
            'speaker_match_score': (cos_sim + 1) / 2
        }
