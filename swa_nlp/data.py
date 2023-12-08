import re
import string
from pandas import DataFrame

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import nltk.corpus

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('wordnet') 
from nltk import WordNetLemmatizer

lem = WordNetLemmatizer()


class SwahiliTextClassificationDataset(Dataset):
    """Data class to get samples"""
    
    def __init__(
        self,
        data: DataFrame,
        tokenizer: PreTrainedTokenizer,):
        """
        Initialises the dataset class

        Args:
            data (DataFrame): 
                dataframe containing sentences that constitutes the data and optionally train labels.
            tokenizer (PreTrainedTokenizer): 
                Hugging-Face tokenizer use to tokenize the data.
        """
        
        # setup up the data samples
        self.dataset_ids = data['id'].to_list()
        sentences = [self.preprocess(text) for text in data['content'].to_list()]
        self.encodings = tokenizer(
            sentences, 
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True,
        )
        self.y = None
        
        if 'category' in data.columns:
            # get labels from dataframe
            self.y = data['category'].to_list()

    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.y is not None:
            item['labels'] = torch.tensor(self.y[idx])
        else:
            item['labels'] = torch.tensor(-1)
        
        item['dataset_ids'] = self.dataset_ids[idx]
        return item
    
    def __len__(self):
        return len(self.dataset_ids)
    
    @staticmethod
    def preprocess(text):
        # Transform to lowercase
        text = text.lower()
        
        # Remove hashtags and mentions
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)

        # Remove emojis
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Remove URL links
        text = re.sub(r'http\S+', '', text)
        
        # Remove stop words and punctuations
        stop = set(stopwords.words('english')).union(set(string.punctuation))
        text = " ".join([word for word in text.split() if word not in stop])
        
        # Remove leading and trailing whitespace
        text = text.strip()
        
        # Replace inflected words with base form
        text = " ".join([lem.lemmatize(word) for word in text.split() ])
        
        return text
        
        
        
        