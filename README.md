# DTrOCR
A PyTorch implementation of DTrOCR: Decoder-only Transformer for Optical Character Recognition.

> [!NOTE]
>
> The author of this repository is not in any way affiliated to the author of the [DTrOCR paper](https://doi.org/10.48550/arXiv.2308.15996). This implementation is purely based on the published details of DTrOCR model architecture and its training.

> [!CAUTION]
>
> This is a work in progress!

## Installation

```shell
git clone git@github.com:arvindrajan92/DTrOCR.git
cd DTrOCR
pip install -r requirements.txt
```

## Usage

```python
from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor

from PIL import Image
from dataclasses import asdict

config = DTrOCRConfig()
model = DTrOCRLMHeadModel(config)
processor = DTrOCRProcessor(config=config, add_bos_token=True, add_eos_token=True)

inputs = processor(
    images=[Image.open("RGB", config.image_size[::-1])],
    texts=["This is a sentence"],
    padding=True,
    return_tensors="pt",
    return_labels=True
)
model_output = model(**asdict(inputs))
```

