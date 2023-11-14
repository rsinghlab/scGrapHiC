import torch



class InnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()
        self.act = torch.nn.Sigmoid()
        
        
    def forward(self, z_0, z_1):
        z_1 = torch.transpose(z_1, 1, 2)
        adj = self.act(torch.matmul(z_0, z_1))
        adj = adj.reshape(adj.shape[0], 1, adj.shape[1], adj.shape[2])
        return adj
