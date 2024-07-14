import torch

from typing import Optional
from dataclasses import dataclass


@dataclass
class DTrOCROutput:
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1, )`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
