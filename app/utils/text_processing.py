import string
from typing import List
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

class TextProcessor:
    
    @staticmethod
    def standardize_letter_case(sentence: str) -> str:
        return sentence.lower()
    
    @staticmethod
    def remove_stop_words(words: List[str], language: str = 'english') -> List[str]:
        stop_words = set(stopwords.words(language))
        return [word for word in words if word.lower() not in stop_words]
    
    @staticmethod
    def lemmatize_words(words: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word.lower()) for word in words]
    
    @staticmethod
    def remove_punctuation(words: List[str]) -> List[str]:
        translator = str.maketrans('', '', string.punctuation)
        return [word.translate(translator) for word in words]
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return word_tokenize(text)
    
    def process_text(self, text: str, methods: List[str], language: str = 'english') -> str:
        """Process text with specified methods"""
        words = self.tokenize(text)
        
        if "remove_punctuation" in methods:
            words = self.remove_punctuation(words)
        
        if "standardize_letter_case" in methods:
            words = [self.standardize_letter_case(word) for word in words]
        
        if "remove_stop_words" in methods:
            words = self.remove_stop_words(words, language)
        
        if "lemmatize_each_word" in methods:
            words = self.lemmatize_words(words)
        
        return ' '.join(words)