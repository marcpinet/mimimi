"""
Complex Anomaly Detection for Industrial Sounds
==============================================

A framework for detecting anomalies in industrial machine sounds
using complex-valued neural networks and attention mechanisms.

"""

__version__ = "1.0.0"

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core configuration
try:
    from .config import Config
except ImportError as e:
    logger.error(f"Failed to import config: {e}")
    raise

# Utilities
try:
    from .utils import find_youden_threshold
    logger.info("Utility functions loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import utility functions: {e}")
    find_youden_threshold = None


# Complex layers
try:
    from .complex_layers import (
        ComplexConv2d,
        ComplexBatchNorm2d,
        ComplexPReLU,
        ComplexConvBlock,
        complex_magnitude,
        complex_to_magnitude_phase
    )
    logger.info("Complex layers loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import complex layers: {e}")
    ComplexConv2d = None
    ComplexBatchNorm2d = None
    ComplexPReLU = None
    ComplexConvBlock = None
    complex_magnitude = None
    complex_to_magnitude_phase = None

# Attention mechanisms
try:
    from .attention import AttentionBlock
    logger.info("Attention mechanisms loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import attention: {e}")
    AttentionBlock = None

# Models
try:
    from .model import ComplexAnomalyDetector
    logger.info("Models loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import models: {e}")
    ComplexAnomalyDetector = None

# Datasets
try:
    from .dataset import (
        MIMIIDataset,
    )
    logger.info("Datasets loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import datasets: {e}")
    MIMIIDataset = None

# Training
try:
    from .train import (
        Trainer,
        kl_divergence_loss,
        create_custom_collate_fn
    )
    logger.info("Training pipeline loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import trainer: {e}")
    Trainer = None
    kl_divergence_loss = None
    create_custom_collate_fn = None

# Evaluation
try:
    from .evaluate import (
        Evaluator,
        evaluate_model,
        convert_numpy_types
    )
    logger.info("Evaluation utilities loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import evaluation: {e}")
    Evaluator = None
    evaluate_model = None
    convert_numpy_types = None

def check_dependencies():
    """Check if required dependencies are available."""
    required_deps = [
        'torch', 'torchvision', 'librosa', 'numpy', 
        'sklearn', 'matplotlib', 'tqdm', 'torchmetrics', 'seaborn'
    ]
    
    missing = []
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        logger.warning(f"Missing dependencies: {missing}")
    else:
        logger.info("All dependencies available")
    
    return missing

def quick_train(machine_type, **config_kwargs):
    """
    Quick training function.
    
    Args:
        machine_type (str): 'fan', 'pump', or 'slider'
        **config_kwargs: Config overrides
    
    Returns:
        dict: Training results
    """
    if Trainer is None:
        raise ImportError("Trainer not available")
    
    config = Config()
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    trainer = Trainer(machine_type, config)
    return trainer.train()

def quick_evaluate(machine_type, model_path, **config_kwargs):
    """
    Quick evaluation function.
    
    Args:
        machine_type (str): Type of machine
        model_path (str): Path to model
        **config_kwargs: Config overrides
    
    Returns:
        dict: Evaluation results
    """
    if evaluate_model is None:
        raise ImportError("Evaluation utilities not available")
    
    config = Config()
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return evaluate_model(machine_type, model_path, config)

def get_available_components():
    """Get list of available components."""
    components = {
        'models': [],
        'datasets': [],
        'layers': [],
        'trainers': [],
        'evaluators': []
    }
    
    if ComplexAnomalyDetector is not None:
        components['models'].append('ComplexAnomalyDetector')
    
    if MIMIIDataset is not None:
        components['datasets'].append('MIMIIDataset')
    
    if ComplexConv2d is not None:
        components['layers'].extend(['ComplexConv2d', 'ComplexBatchNorm2d', 'ComplexPReLU', 'ComplexConvBlock'])
    if AttentionBlock is not None:
        components['layers'].append('AttentionBlock')
    
    if Trainer is not None:
        components['trainers'].append('Trainer')
    
    if Evaluator is not None:
        components['evaluators'].append('Evaluator')
    
    return components

def print_system_info():
    """Print system information and available components."""
    print(f"Complex Anomaly Detection Framework v{__version__}")
    print("=" * 50)
    
    # Check dependencies
    deps = check_dependencies()
    print("Dependencies:")
    for dep in ['torch', 'torchvision', 'librosa', 'numpy', 'sklearn', 'matplotlib', 'tqdm', 'torchmetrics', 'seaborn']:
        status = "✓" if dep not in deps else "✗"
        print(f"  {status} {dep}")
    
    print("\nAvailable Components:")
    components = get_available_components()
    for category, items in components.items():
        if items:
            print(f"  {category.title()}:")
            for item in items:
                print(f"    ✓ {item}")

# Main exports
__all__ = [
    # Configuration
    'Config',
    
    # Models
    'ComplexAnomalyDetector',
    
    # Datasets
    'MIMIIDataset',
    
    # Complex layers
    'ComplexConv2d',
    'ComplexBatchNorm2d',
    'ComplexPReLU',
    'ComplexConvBlock',
    'complex_magnitude',
    'complex_to_magnitude_phase',
    
    # Attention
    'AttentionBlock',
    
    # Training
    'Trainer',
    'kl_divergence_loss',
    'create_custom_collate_fn',
    
    # Evaluation
    'Evaluator',
    'evaluate_model',
    'convert_numpy_types',
    
    # Utilities
    'check_dependencies',
    'get_available_components',
    'print_system_info',
    'quick_train',
    'quick_evaluate',
    'find_youden_threshold',
    
    # Version info
    '__version__'
]

# Check dependencies on import
check_dependencies()

logger.info(f"Complex Anomaly Detection Framework v{__version__} initialized")