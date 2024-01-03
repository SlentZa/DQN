import torch
import torch.nn as nn
import torch.autograd as autograd


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim          # (4,)
        self.output_dim = output_dim        # 2

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2, stride=1),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Conv1d(32, 64, kernel_size=2, stride=1),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Conv1d(64, 64, kernel_size=2, stride=1),
            # nn.ReLU(),
            nn.Tanh()
        )

        self.fc_input_dim = self.feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(128, 256),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1).permute(1, 0)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, 1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals
