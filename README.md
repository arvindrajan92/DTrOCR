# DTrOCR
A PyTorch implementation of DTrOCR: Decoder-only Transformer for Optical Character Recognition.

> [!NOTE]
>
> The author of this repository is not in any way affiliated to the author of the [DTrOCR paper](https://doi.org/10.48550/arXiv.2308.15996). This implementation is purely based on the published details of DTrOCR model architecture and its training.

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

config = DTrOCRConfig()
model = DTrOCRLMHeadModel(config)
processor = DTrOCRProcessor(DTrOCRConfig())

path_to_image = ""  # path to image file

inputs = processor(
    images=Image.open(path_to_image).convert('RGB'),
    texts=processor.tokeniser.bos_token,
    return_tensors="pt"
)

model_output = model.generate(
    inputs=inputs, 
    processor=processor, 
    num_beams=3  # defaults to 1 if not specified
)

predicted_text = processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
```

