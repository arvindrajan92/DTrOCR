from dtrocr.config import DTrOCRConfig
from dtrocr.model import DTrOCRLMHeadModel
from dtrocr.processor import DTrOCRProcessor

import time
import torch
import random
from PIL import Image
from dataclasses import asdict

# global variables
BATCH_SIZE = random.choice(range(1, 10))
BEAM_SIZE = random.choice(range(1, 3))
CONFIG = DTrOCRConfig()
MODEL = DTrOCRLMHeadModel(CONFIG)

# set model to evaluation mode
MODEL.eval()


def test_model():
    processor = DTrOCRProcessor(config=CONFIG, add_bos_token=True, add_eos_token=True)

    inputs = processor(
        images=[Image.new("RGB", CONFIG.image_size[::-1]) for _ in range(BATCH_SIZE)],
        texts=["This is a sentence" for _ in range(BATCH_SIZE)],
        padding=True,
        return_tensors="pt",
        return_labels=True
    )

    model_output = MODEL(**asdict(inputs))

    assert model_output.loss.shape == ()
    assert model_output.accuracy.shape == ()
    assert model_output.logits.shape == (
        BATCH_SIZE,
        int(((CONFIG.image_size[0] / CONFIG.patch_size[0]) * (CONFIG.image_size[1] / CONFIG.patch_size[1]))) +
        inputs.attention_mask.shape[1],
        CONFIG.vocab_size
    )


def test_generation_to_be_deterministic():
    processor = DTrOCRProcessor(DTrOCRConfig())

    inputs = processor(
        images=[Image.new("RGB", CONFIG.image_size[::-1]) for _ in range(BATCH_SIZE)],
        texts=[processor.tokeniser.bos_token for _ in range(BATCH_SIZE)],
        return_tensors="pt"
    )

    output_1 = MODEL.generate(inputs=inputs, processor=processor, num_beams=BEAM_SIZE, use_cache=False)
    output_2 = MODEL.generate(inputs=inputs, processor=processor, num_beams=BEAM_SIZE, use_cache=False)
    assert torch.equal(output_1, output_2)


def test_generation_with_and_without_caching():
    processor = DTrOCRProcessor(DTrOCRConfig())

    inputs = processor(
        images=[Image.new("RGB", CONFIG.image_size[::-1]) for _ in range(BATCH_SIZE)],
        texts=[processor.tokeniser.bos_token for _ in range(BATCH_SIZE)],
        return_tensors="pt"
    )

    start_time = time.time()
    output_without_cache = MODEL.generate(inputs=inputs, processor=processor, num_beams=BEAM_SIZE, use_cache=False)
    time_without_cache = time.time() - start_time

    start_time = time.time()
    output_with_cache = MODEL.generate(inputs=inputs, processor=processor, num_beams=BEAM_SIZE, use_cache=True)
    time_with_cache = time.time() - start_time

    assert torch.equal(output_without_cache, output_with_cache)
    assert time_with_cache <= time_without_cache
