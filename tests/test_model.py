from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor

import time
import torch
import random
from PIL import Image
from dataclasses import asdict


def test_model():
    batch_size = random.choice(range(1, 10))

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
    assert model_output.accuracy.shape == ()
    assert model_output.logits.shape == (
        batch_size,
        int(((config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1]))) +
        inputs.attention_mask.shape[1],
        config.vocab_size
    )


def test_generation_to_be_deterministic():
    beam_size = random.choice(range(1, 3))

    config = DTrOCRConfig()
    model = DTrOCRLMHeadModel(config)
    processor = DTrOCRProcessor(DTrOCRConfig())

    inputs = processor(
        images=Image.new("RGB", config.image_size[::-1]),
        texts=processor.tokeniser.bos_token,
        return_tensors="pt"
    )

    model.eval()

    output_1 = model.generate(inputs=inputs, processor=processor, num_beams=beam_size, use_cache=False)
    output_2 = model.generate(inputs=inputs, processor=processor, num_beams=beam_size, use_cache=False)
    assert torch.equal(output_1, output_2)


def test_generation_with_and_without_caching():
    beam_size = random.choice(range(1, 3))

    config = DTrOCRConfig()
    model = DTrOCRLMHeadModel(config)
    processor = DTrOCRProcessor(DTrOCRConfig())

    inputs = processor(
        images=Image.new("RGB", config.image_size[::-1]),
        texts=processor.tokeniser.bos_token,
        return_tensors="pt"
    )

    model.eval()

    start_time = time.time()
    output_without_cache = model.generate(inputs=inputs, processor=processor, num_beams=beam_size, use_cache=False)
    time_without_cache = time.time() - start_time

    start_time = time.time()
    output_with_cache = model.generate(inputs=inputs, processor=processor, num_beams=beam_size, use_cache=True)
    time_with_cache = time.time() - start_time

    assert torch.equal(output_without_cache, output_with_cache)
    assert time_with_cache <= time_without_cache
