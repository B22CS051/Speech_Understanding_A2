"""
Language Identification (LID) Module
Implements frame-level LID for Hindi vs English code-switched speech
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from scipy import signal


class LIDFeatureExtractor:
    """Extract frame-level features for LID"""
    
    def __init__(self, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_mfcc(self, audio):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc, 
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        # Normalize
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
        return mfcc.T  # (n_frames, n_mfcc)
    
    def extract_spectral_features(self, audio):
        """Extract spectral features (energy, centroid, rolloff)"""
        # Spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_fft=self.n_fft, 
                                              hop_length=self.hop_length, n_mels=40)
        # Log scale
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = (spec_db - spec_db.mean(axis=1, keepdims=True)) / (spec_db.std(axis=1, keepdims=True) + 1e-9)
        
        # Compute frame-based features
        frame_energy = np.mean(np.abs(spec_db), axis=0)  # (n_frames,)
        
        return spec_db.T, frame_energy  # (n_frames, 40), (n_frames,)
    
    def extract_features(self, audio):
        """Extract full feature set"""
        # Ensure audio is in correct range
        audio = np.asarray(audio, dtype=np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        # Extract different features
        mfcc = self.extract_mfcc(audio)
        mel_spec, energy = self.extract_spectral_features(audio)
        
        # Combine features
        energy = energy.reshape(-1, 1)  # (n_frames, 1)
        features = np.concatenate([mfcc, mel_spec, energy], axis=1)  # (n_frames, 54)
        
        return features


class LIDModel(nn.Module):
    """CNN-LSTM based Language Identification Model"""
    
    def __init__(self, input_dim=54, hidden_dim=128, num_layers=2, num_classes=2):
        super(LIDModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.3)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, 
                           bidirectional=True, dropout=0.3)
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) or (batch_size, input_dim, seq_len)
        Returns:
            logits: (batch_size, seq_len, num_classes)
        """
        # CNN processing
        if x.dim() == 3:
            batch_size, seq_len, feat_dim = x.shape
            # Transpose to (batch_size, feat_dim, seq_len)
            x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x)
        x = self.pool(x)
        
        # Transpose back to (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Frame-level classification
        batch_size, seq_len, _ = lstm_out.shape
        logits = self.fc(lstm_out.reshape(-1, lstm_out.size(-1)))
        logits = logits.reshape(batch_size, seq_len, -1)
        
        return logits


class LanguageIdentifier:
    """Main LID system wrapper"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.feature_extractor = LIDFeatureExtractor()
        self.model = LIDModel().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
        self.classes = {0: 'English', 1: 'Hindi'}
        
    def identify(self, audio, sr=16000):
        """
        Identify language at frame level
        
        Args:
            audio: numpy array or path to audio file
            sr: sample rate
            
        Returns:
            predictions: (n_frames,) array with language IDs
            confidences: (n_frames,) array with confidence scores
            timestamps: (n_frames,) array with frame timestamps
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=sr)
        
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(features)
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
        
        predictions = predictions.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        
        # Compute timestamps
        hop_length = self.feature_extractor.hop_length
        timestamps = (np.arange(len(predictions)) * hop_length) / sr
        
        return predictions, confidences, timestamps
    
    def get_switch_points(self, audio, sr=16000, threshold=0.5):
        """
        Identify code-switching boundaries
        
        Args:
            audio: numpy array or path to audio file
            sr: sample rate
            threshold: confidence threshold for switching
            
        Returns:
            switch_points: list of (timestamp, from_lang, to_lang, confidence)
        """
        predictions, confidences, timestamps = self.identify(audio, sr)
        
        switch_points = []
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                if confidences[i] >= threshold or confidences[i-1] >= threshold:
                    switch_points.append((
                        float(timestamps[i]),
                        self.classes[predictions[i-1]],
                        self.classes[predictions[i]],
                        float(max(confidences[i], confidences[i-1]))
                    ))
        
        return switch_points


