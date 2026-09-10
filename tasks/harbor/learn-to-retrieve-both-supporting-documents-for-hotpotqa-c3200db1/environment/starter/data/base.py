from abc import ABC, abstractmethod
from typing import Generator, Tuple, Any
import jax.numpy as jnp
import os

class BaseDataset(ABC):
    """
    Abstract base class for all dataset implementations.
    Ensures compatibility with training and evaluation scripts.
    """
    
    def __init__(
        self, 
        tokenizer=None, 
        split: str = "train", 
        limit: int = None, 
        seq_len: int = 256, 
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 8,
        mask_prefix: bool = True,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer object (e.g., from HuggingFace)
            split: Dataset split to load (e.g., "train", "test", "validation")
            limit: Maximum number of samples to process
            seq_len: Maximum sequence length for tokenization
            batch_size: Number of samples per batch
            **kwargs: Additional dataset-specific configurations
        """
        self.tokenizer = tokenizer
        self.split = split
        self.limit = limit
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.mask_prefix = mask_prefix
        self.kwargs = kwargs
        
        self.hf_token = os.environ.get("HF_TOKEN")
        if self.hf_token is None:
            raise ValueError("HF_TOKEN environment variable is not set")
        

    @abstractmethod
    def generator(self) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
        """
        Generator that yields batches of tokenized data.
        
        Yields:
            Tuple of (batch_tokens, batch_masks)
            - batch_tokens: jnp.ndarray of shape (batch_size, seq_len + 1)
            - batch_masks: jnp.ndarray of shape (batch_size, seq_len + 1)
        """
        pass


