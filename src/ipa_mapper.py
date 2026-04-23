"""
IPA Mapper Module
Converts code-switched Hinglish text to IPA representation
"""

import re
from typing import Dict, List, Tuple


class HinglishIPAMapper:
    """Maps Hinglish graphemes to IPA phonemes"""
    
    def __init__(self):
        # English phoneme mappings
        self.english_mappings = {
            # Vowels
            'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɔ', 'u': 'ʊ',
            'ai': 'aɪ', 'ay': 'eɪ', 'oi': 'ɔɪ', 'ou': 'oʊ', 'ow': 'aʊ',
            'aa': 'ɑ', 'ah': 'ɑ', 'ar': 'ɑɹ',
            'er': 'ɹ', 'ir': 'ɪɹ', 'ur': 'ɝ', 'or': 'ɔɹ',
            
            # Consonants
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': 'h', 'j': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'p': 'p', 'r': 'ɹ', 's': 's', 't': 't',
            'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z',
            'ch': 'tʃ', 'sh': 'ʃ', 'th': 'θ',
            'ng': 'ŋ', 'gh': 'g', 'ph': 'f',
            'qu': 'kw', 'ck': 'k',
        }
        
        # Hindi phoneme mappings (Hinglish romanization to IPA)
        self.hindi_mappings = {
            # Vowels
            'a': 'ə', 'aa': 'ɑ', 'i': 'i', 'ii': 'iː',
            'u': 'u', 'uu': 'uː', 'e': 'e', 'ai': 'ɛ', 'o': 'o', 'au': 'ɔ',
            'aw': 'ɔ',
            
            # Consonants
            'k': 'k', 'kh': 'kʰ', 'g': 'g', 'gh': 'gʰ', 'ng': 'ŋ',
            'ch': 'tʃ', 'chh': 'tʃʰ', 'j': 'dʒ', 'jh': 'dʒʰ', 'n': 'n',
            't': 't̪', 'th': 't̪ʰ', 'd': 'd̪', 'dh': 'd̪ʰ', 'nn': 'ɳ',
            'p': 'p', 'ph': 'pʰ', 'b': 'b', 'bh': 'bʰ', 'm': 'm',
            'y': 'j', 'r': 'ɾ', 'l': 'l', 'w': 'ʋ', 'v': 'ʋ',
            's': 's', 'sh': 'ʃ', 'sh': 'ʂ', 'h': 'ɦ',
        }
        
        # Common Hinglish words and phrases
        self.hinglish_phrases = {
            'haan': 'ɦɑn', 'nahin': 'nəɦɪ̃',
            'kya': 'kjə', 'wo': 'ʋoː',
            'ho': 'ɦoː', 'raha': 'ɾəɦɑ',
            'matlab': "mət'lɑb", 'samjho': 'səmdʒoː',
            'dekho': 'deːkʰoː', 'suno': 'sunoː',
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for IPA conversion"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def english_to_ipa(self, word: str) -> str:
        """Convert English word to IPA"""
        word = word.lower()
        ipa = ""
        i = 0
        
        while i < len(word):
            matched = False
            
            # Try 2-character sequences first
            if i + 1 < len(word):
                bigram = word[i:i+2]
                if bigram in self.english_mappings:
                    ipa += self.english_mappings[bigram]
                    i += 2
                    matched = True
            
            # Try single character
            if not matched:
                char = word[i]
                if char in self.english_mappings:
                    ipa += self.english_mappings[char]
                else:
                    ipa += char  # Keep unknown characters
                i += 1
        
        return ipa
    
    def hindi_to_ipa(self, word: str) -> str:
        """Convert Hindi/Hinglish romanized word to IPA"""
        word = word.lower()
        ipa = ""
        i = 0
        
        while i < len(word):
            matched = False
            
            # Try 2-character sequences first
            if i + 1 < len(word):
                bigram = word[i:i+2]
                if bigram in self.hindi_mappings:
                    ipa += self.hindi_mappings[bigram]
                    i += 2
                    matched = True
            
            # Try single character
            if not matched:
                char = word[i]
                if char in self.hindi_mappings:
                    ipa += self.hindi_mappings[char]
                else:
                    ipa += char
                i += 1
        
        return ipa
    
    def detect_language(self, word: str) -> str:
        """Simple heuristic to detect English vs Hindi"""
        hindi_indicators = ['aa', 'ii', 'uu', 'kh', 'gh', 'ch', 'sh', 'th']
        
        word_lower = word.lower()
        hindi_score = sum(word_lower.count(ind) for ind in hindi_indicators)
        
        # Check if word is in Hinglish phrases
        if word_lower in self.hinglish_phrases:
            return 'mixed'
        
        return 'hindi' if hindi_score > 0 else 'english'
    
    def text_to_ipa(self, text: str) -> str:
        """Convert full text to IPA"""
        text = self.preprocess_text(text)
        words = text.split()
        
        ipa_words = []
        for word in words:
            language = self.detect_language(word)
            
            if word.lower() in self.hinglish_phrases:
                ipa = self.hinglish_phrases[word.lower()]
            elif language == 'hindi':
                ipa = self.hindi_to_ipa(word)
            else:
                ipa = self.english_to_ipa(word)
            
            ipa_words.append(ipa)
        
        return ' '.join(ipa_words)
    
    def get_phoneme_sequence(self, text: str) -> List[str]:
        """Get sequence of phonemes with word boundaries"""
        text = self.preprocess_text(text)
        words = text.split()
        
        phoneme_seq = []
        for word in words:
            if word.lower() in self.hinglish_phrases:
                ipa = self.hinglish_phrases[word.lower()]
            else:
                language = self.detect_language(word)
                if language == 'hindi':
                    ipa = self.hindi_to_ipa(word)
                else:
                    ipa = self.english_to_ipa(word)
            
            # Split into individual phonemes
            phonemes = list(ipa)
            phoneme_seq.extend(phonemes)
            phoneme_seq.append('#')  # Word boundary marker
        
        return phoneme_seq
    
    def add_custom_mapping(self, word: str, ipa: str, language: str = 'mixed'):
        """Add custom word-level mappings"""
        self.hinglish_phrases[word.lower()] = ipa


class IPAPhonemeAligner:
    """Align phonemes with audio frames"""
    
    def __init__(self, hop_length=512, sr=16000):
        self.hop_length = hop_length
        self.sr = sr
        self.frame_duration = hop_length / sr
    
    def align_phonemes(self, phoneme_sequence: List[str], 
                      duration: float) -> List[Tuple[str, float, float]]:
        """
        Align phoneme sequence to audio duration
        
        Args:
            phoneme_sequence: list of phonemes
            duration: audio duration in seconds
            
        Returns:
            list of (phoneme, start_time, end_time)
        """
        num_frames = int(duration * self.sr / self.hop_length)
        frames_per_phoneme = num_frames / len(phoneme_sequence)
        
        aligned = []
        for idx, phoneme in enumerate(phoneme_sequence):
            start = idx * frames_per_phoneme * self.frame_duration
            end = (idx + 1) * frames_per_phoneme * self.frame_duration
            aligned.append((phoneme, start, end))
        
        return aligned


class CodeSwitchIPAMapper:
    """Handle code-switched text with language labels"""
    
    def __init__(self):
        self.mapper = HinglishIPAMapper()
    
    def convert_transcript_to_ipa(self, segments: List[Dict]) -> List[Dict]:
        """
        Convert transcript segments with language info to IPA
        
        Args:
            segments: list of dicts with 'text' and 'language' keys
            
        Returns:
            list of dicts with IPA representations added
        """
        enriched = []
        for seg in segments:
            text = seg.get('text', '')
            ipa = self.mapper.text_to_ipa(text)
            
            enriched.append({
                **seg,
                'ipa': ipa,
                'phonemes': self.mapper.get_phoneme_sequence(text)
            })
        
        return enriched
