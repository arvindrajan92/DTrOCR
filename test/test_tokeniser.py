from dtrocr.processor import DTrOCRProcessor
from dtrocr.config import DTrOCRConfig


def test_tokeniser_with_bos_token():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(), add_bos_token=True)
    tokeniser_output = tokeniser(texts=["This is a sentence", "That is not a sentence, sorry"])

    expected_input_ids = [
        [50256, 1212, 318, 257, 6827],
        [50256, 2504, 318, 407, 257, 6827, 11, 7926]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_tokeniser_with_eos_token():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(), add_eos_token=True)
    tokeniser_output = tokeniser(texts=["This is a sentence", "That is not a sentence, sorry"])

    expected_input_ids = [
        [1212, 318, 257, 6827, 50256],
        [2504, 318, 407, 257, 6827, 11, 7926, 50256]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask


def test_tokeniser_with_eos_and_bos_tokens():
    tokeniser = DTrOCRProcessor(config=DTrOCRConfig(), add_bos_token=True, add_eos_token=True)
    tokeniser_output = tokeniser(texts=["This is a sentence", "That is not a sentence, sorry"])

    expected_input_ids = [
        [50256, 1212, 318, 257, 6827, 50256],
        [50256, 2504, 318, 407, 257, 6827, 11, 7926, 50256]
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    assert tokeniser_output.input_ids == expected_input_ids
    assert tokeniser_output.attention_mask == expected_attention_mask
