import torch
import torch.nn as nn


# class ModelConfig:
#
#     def __init__(self,
#                  input_size: int = 1,
#                  hidden_size: int = 64,
#                  num_layers: int = 2,
#                  output_size: int = 1,
#                  dropout_rate: float = 0.1):
#         self.input_size   = input_size
#         self.hidden_size  = hidden_size
#         self.num_layers   = num_layers
#         self.output_size  = output_size
#         self.dropout_rate = dropout_rate


class LSTMCNN(nn.Module):
    def __init__(self, cfg):
        super(LSTMCNN, self).__init__()
        # Copy model structure params from cfg
        self.input_size   = cfg.input_size
        self.hidden_size  = cfg.hidden_size
        self.num_layers   = cfg.num_layers
        self.output_size  = cfg.output_size
        self.dropout_rate = cfg.dropout_rate

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0
        )

        # Final fully‐connected layer
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.to(cfg.device)

    def forward(self, x):
        # x: [B, 1, H, W] → reshape to [B, L, 1]
        B = x.size(0)
        x = x.view(B, -1, 1)           # [B, L, 1]
        x = x.permute(0, 2, 1)         # [B, 1, L]
        x = self.cnn(x)                # [B, 512, L]
        x = x.permute(0, 2, 1)         # [B, L, 512]

        # init LSTM states on the same device
        h0 = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, B, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))  # [B, L, hidden_size*2]

        out = self.fc(out)              # [B, L, output_size]
        return out
