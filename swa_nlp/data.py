import numpy as np
from pandas import DataFrame

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


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
        self.dataset_idx = data['id'].to_list()
        sentences = [self.preprocess(text) for text in data['content'].to_list()]
        self.encodings = tokenizer(sentences, truncation=True, padding=True)
        self.y = None
        
        if 'content' in data.columns:
            # get labels from dataframe
            self.y = data['category'].to_list()

    def __getitem__(self, idx):
        item['dataset_idx'] = self.dataset_idx[idx]
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.y is not None:
            item['labels'] = torch.tensor(self.y[idx])
        else:
            item['labels'] = None
            
        return item
    
    def __len__(self):
        return len(self.dataset_idx)
    
    @staticmethod
    def preprocess(text):
        return text
        
        
        