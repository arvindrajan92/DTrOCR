# DTrOCR
![logo](logo.png)
[![Python application](https://github.com/arvindrajan92/DTrOCR/actions/workflows/python-app.yml/badge.svg)](https://github.com/arvindrajan92/DTrOCR/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/arvindrajan92/DTrOCR/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/arvindrajan92/DTrOCR/actions/workflows/github-code-scanning/codeql)
[![Python Versions](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/arvindrajan92/DTrOCR.svg)](https://github.com/arvindrajan92/DTrOCR/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/arvindrajan92/DTrOCR?style=social)](https://github.com/arvindrajan92/DTrOCR)

A PyTorch implementation of DTrOCR: Decoder-only Transformer for Optical Character Recognition.

> [!NOTE]
>
> The author of this repository is not in any way affiliated to the author of the [DTrOCR paper](https://doi.org/10.48550/arXiv.2308.15996). This implementation is purely based on the published details of DTrOCR model architecture and its training.
> 
> Pre-trained weight for the model is not available at this time as this is a personal project with limited resources.

Below are the key differences between the original implementation (from the paper) and this implementation.

|                                                              | Original implementation      | This implementation   |
| ------------------------------------------------------------ | ---------------------------- | --------------------- |
| Maximum token length<br />(including 128 image patch tokens) | 512                          | 256                   |
| Language                                                     | English & Chinese            | English               |
| Pre-training corpus (planned)                                | Scene, printed & handwritten | Printed & handwritten |

## Installation

```shell
git clone https://github.com/arvindrajan92/DTrOCR
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

model.eval()        # set model to evaluation mode for deterministic behaviour
path_to_image = ""  # path to image file

inputs = processor(
    images=Image.open(path_to_image).convert('RGB'),
    texts=processor.tokeniser.bos_token,
    return_tensors="pt"
)

model_output = model.generate(
    inputs=inputs, 
    processor=processor, 
    num_beams=3,    # defaults to 1 if not specified
    use_cache=True  # defaults to True if not specified
)

predicted_text = processor.tokeniser.decode(model_output[0], skip_special_tokens=True)
```

