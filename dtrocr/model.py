import torch

from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings

from typing import Optional, Union, Tuple, List


class DTrOCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embeddings
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.hidden_layers = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        patch_embeddings = self.patch_embeddings(pixel_values)
        token_embeddings = self.token_embedding(input_ids)
        patch_and_token_embeddings = torch.concat([patch_embeddings, token_embeddings], dim=-2)

        input_shape = patch_and_token_embeddings.shape
        position_embeddings = self.positional_embedding(self.position_ids[:, :input_shape[1]])

        hidden_states = patch_and_token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # build causal self attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape[0], patch_and_token_embeddings.shape[-2], dtype=torch.int64)
        else:
            attention_mask = torch.concat(
                [
                    torch.ones(input_shape[0], patch_embeddings.shape[-2], dtype=attention_mask.dtype),
                    attention_mask
                ], dim=-1
            )
        attention_mask = prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_shape[0], input_shape[-2]),
            inputs_embeds=patch_and_token_embeddings,
        )

        # causal self attention must always have access to image patch embeddings
        attention_mask[:, :, :, :patch_embeddings.shape[-2]] = 0

        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


# Adapted from _prepare_4d_causal_attention_mask_for_sdpa in transformers.modeling_attn_mask_utils
def prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.

    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True)

    is_tracing = (
        torch.jit.is_tracing()
        or isinstance(inputs_embeds, torch.fx.Proxy)
        or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
    )

    if attention_mask is None:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], input_shape[-1], dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    else:
        if attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            expanded_4d_mask = attention_mask
        else:
            expanded_4d_mask = attn_mask_converter.to_4d(
                attention_mask,
                input_shape[-1],
                dtype=inputs_embeds.dtype,
                key_value_length=input_shape[-1],
            )

        # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        if not is_tracing and expanded_4d_mask.device.type == "cuda":
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, min_dtype=torch.finfo(inputs_embeds.dtype).min
            )

    return expanded_4d_mask
