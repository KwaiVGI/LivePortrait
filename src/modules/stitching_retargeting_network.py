# coding: utf-8

"""
Stitching module(S) and two retargeting modules(R) defined in the paper.

- The stitching module pastes the animated portrait back into the original image space without pixel misalignment, such as in
the stitching region.

- The eyes retargeting module is designed to address the issue of incomplete eye closure during cross-id reenactment, especially
when a person with small eyes drives a person with larger eyes.

- The lip retargeting module is designed similarly to the eye retargeting module, and can also normalize the input by ensuring that
the lips are in a closed state, which facilitates better animation driving.
"""
from torch import nn


class StitchingRetargetingNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(StitchingRetargetingNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp = nn.Sequential(*layers)

    def initialize_weights_to_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)
