import torch

import lightning.pytorch as pl
import torch.nn.functional as F


from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_dense_batch
from torch.nn import Conv1d, LeakyReLU, BatchNorm1d, Linear, ModuleList






class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention [Vaswani et al. NeurIPS 2017].

    Scaled dot-product attention is performed over V, using K as keys and Q as queries.

        MultiHeadAttention(Q, V) = FC(SoftMax(1/√d QKᵀ) V) (concatenated over multiple heads),

    Notes
    -----
    (1) Q, K, V can be of different dimensions. Q and K are projected to dim_a and V to dim_o.
    (2) We assume the last and second last dimensions correspond to the feature (i.e. embedding)
        and token (i.e. words) dimensions respectively.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout_prob=0.1, dim_a=None, dim_o=None, use_alibi=False):
        super().__init__()
        if dim_a is None:
            dim_a = dim_q
        if dim_o is None:
            dim_o = dim_q
        self.dim_a, self.dim_o, self.num_heads, self.use_alibi = dim_a, dim_o, num_heads, use_alibi
        self.fc_q = nn.Linear(dim_q, dim_a, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_a, bias=True)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=True)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)
        if self.use_alibi:
            log_slopes = torch.log(torch.arange(8, 0, -(8 / self.num_heads)))
            self.log_slopes = nn.Parameter(log_slopes.repeat(self.num_heads, 1, 1))

    def forward(self, q, k, v, mask=None):
        """
        Perform multi-head attention with given queries and values.

        Parameters
        ----------
        q: (bsz, tsz, dim_q)
        k: (bsz, tsz, dim_k)
        v: (bsz, tsz, dim_v)
        mask: (bsz, tsz) or (bsz, tsz, tsz), where 1 denotes keep and 0 denotes remove

        Returns
        -------
        O: (bsz, tsz, dim_o)
        """
        bsz, tsz, _ = q.shape
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        q = torch.stack(q.split(self.dim_a // self.num_heads, dim=-1), dim=1)
        k = torch.stack(k.split(self.dim_a // self.num_heads, dim=-1), dim=1)
        v = torch.stack(v.split(self.dim_o // self.num_heads, dim=-1), dim=1)
        a = q @ k.transpose(-1, -2) / self.dim_a ** 0.5
        if self.use_alibi:
            arange = torch.arange(tsz, device=q.device)
            bias = -torch.abs(arange.unsqueeze(-1) - arange.unsqueeze(-2))
            bias = bias.repeat(self.num_heads, 1, 1) * torch.exp2(-torch.exp(self.log_slopes))
            a.add_(bias.unsqueeze(0))
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            a.masked_fill_(mask == 0, -65504)
        a = self.dropout(torch.softmax(a, dim=-1))
        o = self.fc_o(a @ v).transpose(1, 2).flatten(2, 3)
        return o


class PositionwiseFFN(nn.Module):
    """
    Position-wise FFN [Vaswani et al. NeurIPS 2017].
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))



class EncoderBlock(pl.LightningModule):
    """
    Transformer encoder block [Vaswani et al. NeurIPS 2017].

    Note that this is the pre-LN version [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x_ = self.ln1(x)
        x = x + self.dropout(self.attn(x_, x_, x_, mask))
        x_ = self.ln2(x)
        x = x + self.dropout(self.ffn(x_))
        return x


class ChIPSeqProcessor(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super(ChIPSeqProcessor, self).__init__()
        
        if PARAMETERS['positional_encodings']:
            self.number_channels = PARAMETERS['node_features'] + PARAMETERS['pos_encodings_dim']
        else: 
            self.number_channels = PARAMETERS['node_features']

        self.PARAMETERS = PARAMETERS
        # Feature extractor
        self.feature_extractor = Conv1d(
            in_channels=self.number_channels,
            out_channels=PARAMETERS['encoder_hidden_embedding_size'],
            kernel_size=PARAMETERS['conv1d_kernel_size'],
            padding='same'
        )
        
        self.lr_0 = LeakyReLU()
        self.act = nn.Sigmoid()
        
        self.batch_norm_0 = BatchNorm1d(PARAMETERS['encoder_hidden_embedding_size'])
        
        # Self attention to learn the relationships between the features conditional on their positioning (in the prior)
        self.encoder_blocks = nn.ModuleList()
        for _ in range(PARAMETERS['num_encoder_attn_blocks']):
            self.encoder_blocks.append(
                EncoderBlock(
                    dim=PARAMETERS['encoder_hidden_embedding_size'],
                    hidden_dim=self.number_channels,
                    num_heads=PARAMETERS['num_heads_encoder_attn_blocks']
                )
            )
            

    def forward(self, x, pe):
        # Feature extraction phase
        if len(x.shape) > 3:
            print("Bad input to the Encoder with shape {}".format(x.shape))
            exit(1) # Bad input
        
        if self.PARAMETERS['positional_encodings']:
            x = torch.cat((x, pe), dim=2)
        
        x = x.permute(0, 2, 1)
        x = self.batch_norm_0(self.lr_0(self.feature_extractor(x)))
        x = x.permute(0, 2, 1)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        return self.act(x)


class GraphConvolution(pl.LightningModule):
    def __init__(self, channels, heads, edge_dim):
        super().__init__()

        self.conv = GATv2Conv(
            in_channels = channels,
            out_channels = channels,
            heads = heads, 
            edge_dim = edge_dim,
        )
        self.linear = Linear(channels, channels)
        self.bn = GraphNorm(channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x_ = self.conv(x, edge_index, edge_attr)
        x_ = F.relu(self.linear(x))
        x_ = self.bn(x, batch)
        return x + x_


########################################################## ENCODER ###################################################################################

class GraphEncoder(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.PARAMETERS = PARAMETERS
        
        if self.PARAMETERS['positional_encodings']:
            self.node_emb = Linear(PARAMETERS['encoder_hidden_embedding_size'], PARAMETERS['encoder_hidden_embedding_size'] - PARAMETERS['pos_encodings_dim'])
        else:
            self.node_emb = Linear(PARAMETERS['encoder_hidden_embedding_size'], PARAMETERS['encoder_hidden_embedding_size'])
        
        self.pe_lin = Linear(PARAMETERS['pos_encodings_dim'], PARAMETERS['pos_encodings_dim'])
        self.x_norm = BatchNorm1d(PARAMETERS['encoder_hidden_embedding_size'])
        self.pe_norm = BatchNorm1d(PARAMETERS['pos_encodings_dim'])
        self.channels = PARAMETERS['encoder_hidden_embedding_size']
        
        self.convs = ModuleList()

        for _ in range(PARAMETERS['num_graph_conv_blocks']):
            conv = GraphConvolution(
                self.channels, 
                1, PARAMETERS['edge_dims']
            )
            self.convs.append(conv)

        # Transformer encoder blocks to capture long range dependencies     
        self.encoder_blocks = ModuleList()
        for _ in range(PARAMETERS['num_graph_encoder_blocks']):
            self.encoder_blocks.append(
                EncoderBlock(
                    dim=self.channels,
                    hidden_dim=self.channels,
                    num_heads=PARAMETERS['num_heads_encoder_attn_blocks']
                )
            )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x = self.x_norm(x)
        
        if self.PARAMETERS['positional_encodings']:
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)

        if self.PARAMETERS['use_bulk']:
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr, batch)
        
        x, _ = to_dense_batch(x, batch)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        return x




