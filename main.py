import random
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Swahili Text Classification Training and Evaluation')
    parser.add_argument('--model_name', type=str, help='Timm model name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for testing')
    args = parser.parse_args()
    return args
    

def main():
    
    # get training and submission parameters
    args = parse_args()
    
    # ensure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    
    

if __name__ == "__main__":
    print("Hello World!")