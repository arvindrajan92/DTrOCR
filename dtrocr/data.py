import torch
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union, List


@dataclass
class DTrOCROutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class DTrOCRProcessorOutput:
    pixel_values: Optional[torch.FloatTensor] = None
    input_ids: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    attention_mask: Optional[Union[torch.FloatTensor, np.ndarray, List[int]]] = None
    labels: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
