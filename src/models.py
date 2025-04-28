import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int,
                 output_size: int = 1, dropout_prob: float = 0.3):
        super(SimpleNN, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_prob)]

        # Hidden layers
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # Output layer (no dropout here)
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SimpleNNWithBatchNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int,
                 output_size: int = 1, dropout_prob: float = 0.3):
        super(SimpleNNWithBatchNorm, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_prob)]

        # Hidden layers
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_prob))

        # Output layer (no dropout here)
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
