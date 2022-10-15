from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len : int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_len = max_len

        #print("embedding shape:", embeddings.shape)  ## [4117, 300]

        self.lstm = nn.LSTM(input_size = embeddings.shape[1], hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional, batch_first=True)
        self.ln = nn.LayerNorm(embeddings.shape[1])
        self.bn = nn.BatchNorm1d(max_len)
        self.act = nn.ReLU()

        self.intent_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size * self.max_len, num_class),
        )

        self.slot_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size, num_class),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        else:
            return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # [128, 32]
        batch = self.embed(batch)
        # [128, 32, 300]
        batch, _ = self.lstm(batch)
        # [128, 32, 256]
        batch = batch.resize(batch.size(0), self.encoder_output_size * self.max_len)
        # [128, 8192]
        batch = self.intent_classifier(batch)
        # [128, 150]
       
        return batch


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # [128, 128]
        batch = self.embed(batch)
        # [128, 128, 300]
        batch = self.ln(batch)
        # [128, 128, 300]
        batch, _ = self.lstm(batch)
        # [128, 128, 1024]
        batch = self.bn(batch)
        # [128, 128, 1024]
        batch = self.act(batch)
        # [128, 128, 1024]    
        batch = self.slot_classifier(batch)
        # [128, 128, 9]
        batch = batch.permute([0, 2, 1])

        return batch



