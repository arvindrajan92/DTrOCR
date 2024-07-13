import torch

from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings


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

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor
    ):
        device = input_ids.device if input_ids is not None else pixel_values.device
        past_key_values = tuple([None] * len(self.hidden_layers))

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

        patch_embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=False)
        inputs_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.positional_embedding(position_ids)

        hidden_states = torch.concat([patch_embeddings, inputs_embeddings]) + position_embeddings
        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # Attention mask.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(batch_size, input_shape[-1]),
            inputs_embeds=inputs_embeddings,
            past_key_values_length=0,
        )

        for i, (hidden_layer, past_layer) in enumerate(zip(self.hidden_layers, past_key_values)):
            outputs = hidden_layer(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states
