import torch

import lightning.pytorch as pl
import torch.nn.functional as F

from src.models.Encoder import EncoderBlock

class ABDecoder(pl.LightningModule):
    def __init__(
        self, PARAMETERS
    ):
        super(ABDecoder, self).__init__()
        # Self attention to learn the relationships between the features conditional on their positioning as well
        self.decoder_blocks = torch.nn.ModuleList()
        self.number_channels = PARAMETERS['encoder_hidden_embedding_size']
        
        self.relu = torch.nn.ReLU()
        
        
        for _ in range(PARAMETERS['num_decoder_residual_blocks']):
            self.decoder_blocks.append(
                EncoderBlock(
                    dim=PARAMETERS['encoder_hidden_embedding_size'],
                    hidden_dim=self.number_channels,
                    num_heads=PARAMETERS['num_heads_encoder_attn_blocks']
                )
            )
        
        self.fc1 = torch.nn.Linear(PARAMETERS['encoder_hidden_embedding_size'], PARAMETERS['encoder_hidden_embedding_size']//2, bias=True)
        self.fc2 = torch.nn.Linear(PARAMETERS['encoder_hidden_embedding_size']//2, 1, bias=True)
        
        
    def forward(self, x):
        for decoder_block in self.decoder_blocks:
            x = self.relu(decoder_block(x))
        return torch.nn.functional.tanh(self.fc2(F.relu(self.fc1(x))))

  