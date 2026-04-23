import numpy as np
import librosa
from scipy.io.wavfile import write


class TTSGenerator:
    def __init__(self, sr=22050):
        self.sr = sr

    def synthesize(self, text, embedding, duration=5.0):
        print("[TTS] Generating speech...")

        duration = float(max(1.0, duration))
        num_samples = int(self.sr * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Keep pitch in a clearly audible human-like range.
        base_freq = 120.0 + float(np.mean(embedding)) * 2.0
        base_freq = float(np.clip(base_freq, 90.0, 240.0))

        audio = (
            0.5 * np.sin(2 * np.pi * base_freq * t) +
            0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
        )

        # Smooth full-length envelope (avoids near-silent long tails).
        envelope = np.sin(np.linspace(0, np.pi, num_samples))
        audio = audio * envelope

        audio = 0.95 * (audio / (np.max(np.abs(audio)) + 1e-9))

        return audio.astype(np.float32)


class VoiceCloningPipeline:
    def __init__(self, sr=22050):
        self.sr = sr
        self.tts = TTSGenerator(sr)
        self.reference_embedding = None

    def set_reference_voice(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)

        # simple embedding (no heavy model, safe for assignment)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
        self.reference_embedding = np.mean(mfcc, axis=1)

        print("[TTS] Reference voice set")
        return self.reference_embedding

    def generate_speech(self, text, duration=None):
        if self.reference_embedding is None:
            raise ValueError("Reference voice not set")

        if duration is None:
            duration = max(3.0, len(text.split()) * 0.4)

        audio = self.tts.synthesize(
            text=text,
            embedding=self.reference_embedding,
            duration=duration
        )

        return audio

    def save_audio(self, audio, output_path):
        write(output_path, self.sr, (audio * 32767).astype(np.int16))
        print("[TTS] Saved ->", output_path)