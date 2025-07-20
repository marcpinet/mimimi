#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'model'))

from model.train import main as train_main
from model.evaluate import main as evaluate_main
from model.config import Config


def main():
    parser = argparse.ArgumentParser(description='Complex Network for Machine Sound Anomaly Detection')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], default='both',
                      help='Mode to run: train, evaluate, or both')
    parser.add_argument('--machine-type', type=str, default=None,
                      help='Specific machine type to train/evaluate (fan, pump, slider, valve)')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to dataset')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to saved model (for evaluation)')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cuda, cpu, auto)')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.data_path:
        config.DATA_PATH = args.data_path
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.device != 'auto':
        config.DEVICE = args.device
    
    if args.machine_type:
        config.MACHINE_TYPES = [args.machine_type]
    
    print("Configuration:")
    print(f"  Data Path: {config.DATA_PATH}")
    print(f"  Machine Types: {config.MACHINE_TYPES}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Mode: {args.mode}")
    
    Path(config.SAVE_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    
    if args.mode in ['train', 'both']:
        print("\nStarting training...")
        train_main(config)

    if args.mode in ['evaluate', 'both']:
        print("\nStarting evaluation...")
        evaluate_main(config)
    
    print("\nCompleted!")


if __name__ == "__main__":
    main()
