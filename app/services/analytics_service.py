import pandas as pd
from typing import List
import nltk
from nltk.corpus import stopwords
from app.core import config

class AnalyticsService:
    
    if config.NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.append(config.NLTK_DATA_DIR)

    def __init__(self):
        self.stop_words_english = set(stopwords.words('english'))
        self.stop_words_indonesian = set(stopwords.words('indonesian'))
        self.stop_words = self.stop_words_english.union(self.stop_words_indonesian)
    
    def process_word_count(self, data: List[tuple], column_name: str, top_n: int = 50) -> str:
        """Process word count from database results"""
        df = pd.DataFrame(data, columns=[column_name])
        
        # Remove stop words
        df[column_name] = df[column_name].apply(
            lambda x: [word.lower() for word in x.split() if word.lower() not in self.stop_words]
        )
        
        # Explode and clean
        df = df.dropna().explode(column_name)
        df = df[df[column_name].str.strip() != '']
        
        # Count frequencies
        word_counts = df[column_name].value_counts().reset_index(name='frequency')
        word_counts.columns = [column_name, 'frequency']
        
        # Get top N
        result = word_counts.head(top_n)
        return result.to_json()