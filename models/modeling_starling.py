import torch
from torch import nn
from transformers import LlamaPreTrainedModel, LlamaModel


class StarlingForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.PAD_ID = 0
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_device(self):
        return self.transformer.device

    def forward(
          self,
          input_ids=None,
          past_key_values=None,
          attention_mask=None,
          position_ids=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = int(input_ids.shape[0])
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1].unsqueeze(0))
        scores = torch.stack(scores)
        return scores
    