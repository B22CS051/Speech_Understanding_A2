"""
Translation Module
Converts Hinglish text to Low-Resource Language (Santhali example)
"""

from typing import Dict, List
import json


class TechnicalTermDictionary:
    """Domain-specific technical term dictionary for translations"""
    
    def __init__(self):
        # English -> Santhali technical terms mapping
        # This is a minimal example; in production, expand this corpus
        self.term_pairs = {
            # Speech processing
            'speech': 'bolae-bhar',
            'audio': 'sadh-rang',
            'sound': 'sadh',
            'noise': 'madhya-sadh',
            'signal': 'sanket',
            'frequency': 'sara-sara-mani',
            'filter': 'chani',
            'spectrogram': 'sadh-chitra',
            'pitch': 'sur',
            'tone': 'rang',
            'prosody': 'bolae-rang',
            
            # Machine learning
            'model': 'praman',
            'training': 'siksha',
            'learning': 'seekhna',
            'neural': 'menti-jal',
            'network': 'jal',
            'layer': 'paṛ',
            'activation': 'jagriti',
            'gradient': 'dhal',
            'optimization': 'shuddhi',
            
            # Audio processing
            'mfcc': 'mel-mani-chitra',
            'cepstrum': 'sadh-praman',
            'feature': 'lakshan',
            'extraction': 'nikalna',
            'analysis': 'vikhand',
            'synthesis': 'sanchar',
            
            # General
            'algorithm': 'vidhi',
            'classification': 'bhed',
            'recognition': 'pechaan',
            'identification': 'pehchan',
            'transcription': 'leekhan',
            'translation': 'anuvad',
            'language': 'bhasa',
            'english': 'angrezi',
            'hindi': 'hindi',
            'code-switching': 'bhasa-badal',
        }
    
    def translate_term(self, term: str) -> str:
        """Translate individual technical term"""
        term_lower = term.lower()
        return self.term_pairs.get(term_lower, term)
    
    def add_term_pair(self, english: str, santhali: str):
        """Add custom term pair"""
        self.term_pairs[english.lower()] = santhali
    
    def add_from_dict(self, pairs: Dict[str, str]):
        """Add multiple term pairs"""
        self.term_pairs.update({k.lower(): v for k, v in pairs.items()})


class HinglishToSanthaliTranslator:
    """Translate Hinglish text to Santhali"""
    
    def __init__(self):
        self.dictionary = TechnicalTermDictionary()
        
        # Basic Hinglish to Santhali vocabulary
        self.word_mappings = {
            # Common Hinglish words
            'kya': 'enae',
            'hai': 'aha',
            'hain': 'aha',
            'haan': 'ha',
            'nahin': 'naka',
            'ne': 'na',
            'ka': 'ge',
            'ke': 'ge',
            'ki': 'ge',
            'aur': 'aru',
            'ko': 'le',
            'se': 'se',
            'par': 'par',
            'ye': 'eneea',
            'wo': 'oleea',
            'mere': 'ami-ge',
            'aapka': 'apan-ge',
            'iska': 'ene-ge',
            'uska': 'ole-ge',
            
            # Common verbs
            'dekho': 'menakata',
            'suno': 'neolkata',
            'samjho': 'bogolkata',
            'karo': 'arakata',
            'ho': 'aha',
            'raha': 'me-akata',
            'diya': 'diedh',
            'hai': 'aha',
            
            # Question words
            'kyun': 'nokoe',
            'kahan': 'nakhan',
            'kab': 'kaeda',
            'kaun': 'nokoe',
        }
        
    def translate_sentence(self, text: str) -> str:
        """Translate Hinglish sentence to Santhali"""
        words = text.split()
        translated_words = []
        
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?;:')
            word_lower = clean_word.lower()
            
            # Try technical dictionary first
            if word_lower in self.dictionary.term_pairs:
                translation = self.dictionary.translate_term(word_lower)
            # Try basic word mapping
            elif word_lower in self.word_mappings:
                translation = self.word_mappings[word_lower]
            else:
                # For unknown words, keep them as-is (phonetic transcription)
                translation = clean_word
            
            translated_words.append(translation)
        
        return ' '.join(translated_words)
    
    def create_parallel_corpus(self):
        """Create example parallel corpus"""
        corpus = [
            {
                'english': 'speech recognition',
                'hinglish': 'bolae ki pechaan',
                'santhali': 'bolae-bhar ge pechaan'
            },
            {
                'english': 'neural network model',
                'hinglish': 'neural network ka model',
                'santhali': 'menti-jal ge praman'
            },
            {
                'english': 'audio signal processing',
                'hinglish': 'audio sanket ka process',
                'santhali': 'sadh-rang sanket ge vidhi'
            },
            {
                'english': 'feature extraction',
                'hinglish': 'feature nikalna',
                'santhali': 'lakshan nikalna'
            },
        ]
        return corpus


class SemanticTranslator:
    """Handle semantic-level translation with context"""
    
    def __init__(self):
        self.translator = HinglishToSanthaliTranslator()
        self.context_cache = {}
    
    def translate_with_context(self, text: str, context: str = "") -> str:
        """Translate considering context"""
        # For now, use direct translation
        # In production, could use more sophisticated models
        return self.translator.translate_sentence(text)
    
    def batch_translate(self, texts: List[str]) -> List[str]:
        """Translate multiple texts"""
        return [self.translator.translate_sentence(t) for t in texts]
    
    def get_back_translation(self, translated_text: str) -> str:
        """Simple back-translation for validation"""
        # In production, would use a separate model
        return translated_text


class TranslationValidator:
    """Validate translation quality"""
    
    @staticmethod
    def check_term_coverage(text: str, translator: HinglishToSanthaliTranslator) -> float:
        """
        Check how many technical terms were properly translated
        
        Returns:
            coverage ratio (0-1)
        """
        words = text.lower().split()
        translated_count = 0
        
        for word in words:
            word_clean = word.strip('.,!?;:')
            if word_clean in translator.dictionary.term_pairs or \
               word_clean in translator.word_mappings:
                translated_count += 1
        
        return translated_count / len(words) if words else 0.0
    
    @staticmethod
    def compute_similarity(text1: str, text2: str) -> float:
        """Simple lexical similarity metric"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()


class CodeSwitchTranslator:
    """Handle translation of code-switched segments"""
    
    def __init__(self):
        self.translator = HinglishToSanthaliTranslator()
        self.validator = TranslationValidator()
    
    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Translate transcript segments
        
        Args:
            segments: list with 'text' and optionally 'language'
            
        Returns:
            segments with 'translation' field added
        """
        translated = []
        for seg in segments:
            text = seg.get('text', '')
            
            # Translate
            if text:
                translation = self.translator.translate_sentence(text)
            else:
                translation = ""
            
            translated.append({
                **seg,
                'translation': translation,
                'term_coverage': self.validator.check_term_coverage(text, self.translator)
            })
        
        return translated
    
    def build_custom_dictionary(self, technical_terms: Dict[str, str]):
        """Build custom dictionary for specific domain"""
        self.translator.dictionary.add_from_dict(technical_terms)


def load_parallel_corpus(filepath: str) -> List[Dict]:
    """Load parallel corpus from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_parallel_corpus(corpus: List[Dict], filepath: str):
    """Save parallel corpus to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
