import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model, detach_input=False):
        super().__init__()
        self.base_model = base_model
        self.detach_input = detach_input
        hidden_size = base_model.config.n_embd
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        self.reward_head = self.reward_head.to(dtype=torch.bfloat16)

        self._captured_hidden = None
        last_block = self.base_model.transformer.h[-1]
        last_block.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output):
        self._captured_hidden = output

    def forward(self, ids):
        self.base_model(ids)
        last_hidden = self._captured_hidden[:, -1, :]
        if self.detach_input:
            last_hidden = last_hidden.detach()
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
