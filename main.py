import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from swa_nlp.utils import get_dir
from swa_nlp.data import SwahiliTextClassificationDataset
from swa_nlp.model import HuggingFaceTextClassificationModel


def parse_args():
    parser = argparse.ArgumentParser(description='Swahili Text Classification Training and Evaluation')
    parser.add_argument('--model_name', type=str, help='Hugging-Face model name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of training folds for cross-validation')
    args = parser.parse_args()
    return args
    

def main():
    
    # Training and submission parameters
    args = parse_args()
    DATA_DIR =  get_dir('data')
    
    # Setup device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        torch.cuda.manual_seed(args.seed) 
        torch.cuda.manual_seed_all(args.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Use GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data frame from csv
    df_train = pd.read_csv(DATA_DIR / 'Train.csv')
    label_to_idx = {label: idx for idx, label in enumerate(np.unique(df_train['category']))}
    df_train['category'] = df_train['category'].map(lambda x : label_to_idx[x])

    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer == "castorini/afriberta_small":
        # Manually set tokenizer max length as recommended 
        # here https://huggingface.co/castorini/afriberta_small
        tokenizer.model_max_length = 512 
    
    # Start Training & Cross-validation
    for _, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['category'])):
        
        # Get the train and val data for this fold
        train_fold, val_fold = df_train.iloc[train_idx], df_train.iloc[val_idx]
        
        # ==== TRAINING LOOP ==== #
            
        # Initialize Train Dataaset
        train_dataset_fold = SwahiliTextClassificationDataset(
            data=train_fold,
            tokenizer=tokenizer
        )
        
        # Initialize Train DataLoader
        train_loader_fold = DataLoader(
            train_dataset_fold, 
            batch_size=args.train_batch_size, 
            shuffle=True
        )
        
        # Initialize text classification model
        model = HuggingFaceTextClassificationModel(
            model_name=args.model_name,
            num_classes=len(label_to_idx)
        )
        model.to(device)
        
        # Initialize loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # Training
        
        model.train()   
        
        epoch_bar = tqdm(train_loader_fold, unit="batch")
        
        for epoch in range(args.epochs):
            
            for batch in epoch_bar:
                
                epoch_bar.set_description(f"Epoch {epoch}")
                
                # Reset optimizer
                optim.zero_grad()
                
                # Prepare data
                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                
                # Forward pass
                ouput = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Backward pass
                labels = batch['labels'].to(device)
                loss = criterion(ouput, labels)
                loss.backward()
                optimizer.step()
                
                # Logging
                epoch_bar.set_postfix(train_loss=loss.item())
                
        torch.cuda.empty_cache()
        
        # Validation
        
        model.eval()
        
        # Initialize Validation Dataaset
        valid_dataset_fold = SwahiliTextClassificationDataset(
            data=val_fold,
            tokenizer=tokenizer
        )
        
        # Initialize Validation DataLoader
        valid_loader_fold = DataLoader(
            valid_dataset_fold, 
            batch_size=1, 
            shuffle=True
        )
        
        with torch.no_grad():
            
            total_val_loss, total_num = 0.0, 0
            
            for batch in valid_loader_fold:
                
                # Prepare data
                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                
                # Forward pass
                ouput = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Backward pass
                labels = batch['labels'].to(device)
                loss = criterion(ouput, labels)
                
                total_val_loss += loss.item() * len(input_ids)
                total_num += len(input_ids)
            
            epoch_bar.set_postfix(val_loss=total_val_loss / total_num)


if __name__ == "__main__":
    main()