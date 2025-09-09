import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """Calculates a weighted summary of a sequence of LSTM outputs."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, lstm_output):
        weights = self.attention(lstm_output).squeeze(2)
        weights = torch.softmax(weights, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class LandmarkAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.4)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        return x

class Model(nn.Module):
    def __init__(self, num_landmarks, landmark_dim, hidden_size, num_layers, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Linear(landmark_dim, hidden_size)
        self.spatial_attention = LandmarkAttention(hidden_size, num_heads)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        self.temporal_attention = TemporalAttention(hidden_size)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, num_landmarks, landmark_dim = x.shape
        x = x.view(batch_size * seq_len, num_landmarks, landmark_dim)
        x = self.embedding(x)
        x = self.spatial_attention(x)
        x = x.mean(dim=1)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        context_vector = self.temporal_attention(lstm_out)
        out = self.classifier(context_vector)
        return out