from dtrocr.config import DTrOCRConfig
from dtrocr.processor import DTrOCRProcessor
from dtrocr.model import DTrOCRLMHeadModel

import random
from PIL import Image
from dataclasses import asdict


def test_model():
    batch_size = random.choice(range(10))

    config = DTrOCRConfig()
    model = DTrOCRLMHeadModel(config)
    processor = DTrOCRProcessor(config=config, add_bos_token=True, add_eos_token=True)

    inputs = processor(
        images=[Image.new("RGB", config.image_size[::-1]) for _ in range(batch_size)],
        texts=["This is a sentence" for _ in range(batch_size)],
        padding=True,
        return_tensors="pt",
        return_labels=True
    )

    model_output = model(**asdict(inputs))

    assert model_output.loss.shape == ()
    assert model_output.logits.shape == (
        batch_size,
        int(((config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1]))) +
        inputs.attention_mask.shape[1],
        config.vocab_size
    )
