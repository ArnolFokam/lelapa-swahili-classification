from collections import defaultdict
import random
import logging
import argparse
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--num_folds', type=int, default=3, help='Number of training folds for cross-validation')
    args = parser.parse_args()
    return args
    

def main():
    
    logging.basicConfig(
        level=logging.INFO,  # Set logging level to INFO
        format='%(asctime)s [%(levelname)s] %(message)s',  # Define log message format
        handlers=[
            logging.StreamHandler()  # Also log to the console
        ]
    )
    
    # Training and submission parameters
    args = parse_args()
    DATA_DIR =  get_dir('data')
    OUTPUT_DIR = get_dir('./')
    MODEL_NAME="castorini/afriberta_small"
    
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
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        model_max_length=args.max_seq_length
    )
    
    fold_val_losses = []
    models = []
    
    # Start Training & Cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['category'])):
        
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
            model_name=MODEL_NAME,
            num_classes=len(label_to_idx)
        )
        model.to(device)
        
        # Initialize loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # Training
        
        model.train()   
        
        for epoch in range(args.epochs):
            
            total_train_loss, total_num = 0.0, 0
            
            with tqdm(train_loader_fold, unit="batch") as epoch_bar:
            
                for batch in epoch_bar:
                    
                    epoch_bar.set_description(f"Epoch {epoch}")
                    
                    # Reset optimizer
                    optimizer.zero_grad()
                    
                    # Prepare data
                    input_ids = batch['input_ids'].to(device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
                    token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                    
                    # Forward pass
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                    # Backward pass
                    labels = batch['labels'].to(device)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item() * len(input_ids)
                    total_num += len(input_ids)
                    
                    # Logging
                    epoch_bar.set_postfix(train_loss=total_train_loss / total_num)
        
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
            batch_size=args.eval_batch_size, 
            shuffle=True
        )
        
        with torch.no_grad():
            
            total_val_loss, total_num = 0.0, 0
            
            with tqdm(valid_loader_fold, unit="batch") as val_bar:
            
                for batch in val_bar:
                    
                    val_bar.set_description(f"Validation Fold-{fold_idx}")
                    
                    # Prepare data
                    input_ids = batch['input_ids'].to(device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
                    token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                    
                    # Forward pass
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    
                    labels = batch['labels'].to(device)
                    loss = criterion(output, labels)
                    
                    total_val_loss += loss.item() * len(input_ids)
                    total_num += len(input_ids)
                    
                fold_val_losses.append(total_val_loss / total_num)
                logging.info(f"Validation Loss Fold-{fold_idx}: {fold_val_losses[-1]}")
    
        models.append(model)
        
    
    # Logging
    logging.info(f"Mean Validation Loss Fold-{fold_idx}: {sum(fold_val_losses) / len(fold_val_losses)}")
        
        
    # Submission
    
    df_test = df_train = pd.read_csv(DATA_DIR / 'Test.csv')
    df_test['id'] = df_test['swahili_id']
    
    # Initialize Submission Dataaset
    test_dataset = SwahiliTextClassificationDataset(
        data=df_test,
        tokenizer=tokenizer
    )
    
    # Initialize Submission DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size, 
        shuffle=True
    )
    
    predictions = []
    
    with torch.no_grad():
        
        with tqdm(test_loader, unit="batch") as test_bar:
        
            for batch in test_bar:
                
                val_bar.set_description(f"Validation Fold-{fold_idx}")
                
                # Prepare data
                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                dataset_ids = batch['dataset_ids']
                
                preds = []
                    
                for model in models:
                        
                    # Forward pass
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    
                    preds.append(output)
                    
                preds = torch.softmax(torch.stack(preds).mean(dim=0), dim=-1).tolist()
                predictions.extend(list(zip(dataset_ids, preds)))
                
    test_predictions = defaultdict(lambda: [0, 0, 0, 0, 0])
    test_predictions.update(dict(predictions))
    
    # load the sample submission file and update the extent column with the predictions
    submission_df = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')
    
    # ensure the column names matches those from the sample submission
    columns = [text.lower()  for text in label_to_idx.keys()]
    
    # update the extent column with the predictions
    submission_df.loc[:, columns] = submission_df['swahili_id'].map(test_predictions).to_list()
    
    # change id to match expected name
    submission_df = submission_df.rename(columns={"swahili_id": "test_id"})

    # save the submission file and trained model
    submission_df.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
        

if __name__ == "__main__":
    main()