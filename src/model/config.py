import torch

class Config:
    DATA_PATH = '../data'
    MACHINE_TYPES = ['fan', 'pump', 'slider', 'valve']
    
    AUDIO_SR = 16000
    N_FFT = 1024
    HOP_LENGTH = 512
    N_FRAMES = 64
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 30
    PATIENCE = 20
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    SAVE_DIR = '../models'
    LOG_DIR = '../logs'
    
    NORMALIZE = True
    AUGMENT = False
    
    USE_MIXUP = False
    MIXUP_ALPHA = 1.0
    
    TRAIN_NORMAL_ONLY = True
    
    ANOMALY_THRESHOLD = 0.5