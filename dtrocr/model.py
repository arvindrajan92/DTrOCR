import torch

from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings

from typing import Optional


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

        self._attn_implementation = config._attn_implementation

        # initialise GPT-2 weights from Hugging Face
        self.initialise_weights(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        patch_embeddings = self.patch_embeddings(pixel_values)
        token_embeddings = self.token_embedding(input_ids)
        patch_and_token_embeddings = torch.concat([patch_embeddings, token_embeddings], dim=-2)

        input_shape = patch_and_token_embeddings.shape
        position_embeddings = self.positional_embedding(self.position_ids[:, :input_shape[1]])

        hidden_states = patch_and_token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # attention mask
        if attention_mask is not None:
            attention_mask = torch.concat(
                [
                    torch.ones(attention_mask.shape[0], patch_embeddings.shape[-2], dtype=attention_mask.dtype),
                    attention_mask
                ], dim=-1
            )
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(input_shape[0], input_shape[-2]),
                inputs_embeds=patch_and_token_embeddings,
                past_key_values_length=0,
            )

        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states

    def initialise_weights(self, config):
        # load pre-trained GPT-2
        pretrained_gpt2 = GPT2Model.from_pretrained(config.gpt2_hf_model)

        # copy hidden layer weights
        for hidden_layer, pretrained_hidden_layer in zip(self.hidden_layers, pretrained_gpt2.h):
            hidden_layer.load_state_dict(pretrained_hidden_layer.state_dict())

        # token embeddings
        self.token_embedding.load_state_dict(pretrained_gpt2.wte.state_dict())
