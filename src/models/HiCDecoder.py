import torch
import lightning.pytorch as pl



def swish(x):
    return x * torch.sigmoid(x)

class residualBlock(pl.LightningModule):
    def __init__(self, channels, k=3, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        # a swish layer here
        self.conv2 = torch.nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = swish(self.bn1(self.conv1(x)))
        residual =       self.bn2(self.conv2(residual))
        return x + residual



class FullyConnected(pl.LightningModule):

    def __init__(self, embed_dim, hidden_dim, activation=torch.nn.LeakyReLU(0.2)):
        super(FullyConnected, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = torch.nn.Conv2d(self.D, self.H, 1)
        #self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation

    def forward(self, z0, z1):

        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)

        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        
        c = self.conv(z_mul)
        c = self.activation(c)
        
        return c


class HiCDecoder(pl.LightningModule):
    def __init__(
        self, PARAMETERS, activation=torch.nn.Sigmoid()
    ):
        super(HiCDecoder, self).__init__()

        self.hidden = FullyConnected(PARAMETERS['encoder_hidden_embedding_size'], PARAMETERS['encoder_hidden_embedding_size'])
        
        resblocks = [residualBlock(PARAMETERS['encoder_hidden_embedding_size']) for _ in range(PARAMETERS['num_decoder_residual_blocks'])]
        self.resblocks = torch.nn.Sequential(*resblocks)

        self.conv = torch.nn.Conv2d(PARAMETERS['encoder_hidden_embedding_size'], 1, PARAMETERS['width'], padding=PARAMETERS['width'] // 2)
        self.clip()
        
        self.activation = activation
        
    def clip(self):
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        C = self.resblocks(self.cmap(z0, z1))
        return self.predict(C)


    def cmap(self, z0, z1):
        C = self.hidden(z0, z1)
        return C

    def predict(self, C):
        s = self.conv(C)
        s = self.activation(s)
        return s

