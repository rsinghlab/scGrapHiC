
import torch

import numpy as np
import torch.optim as optim
import lightning.pytorch as pl

from src.visualizations import visualize_generated_hic_contact_matrix, visualize_generated_tracks, visualize_hic_contact_matrix
from src.models.Encoder import Encoder
from src.models.HiCDecoder import HiCDecoder
from src.models.ABDecoder import ABDecoder
from src.models.TADDecoder import TADDecoder

class GenomicDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()

        data = np.load(path, allow_pickle=True, mmap_mode='r')
        self.node_features = data['node_features']
        self.hics = data['hic_targets']
        self.abs = data['ab_targets']
        self.tads = data['tad_targets']

    def __getitem__(self, idx: int):
        node_features = self.node_features[idx]# type: ignore
        noise = np.random.normal(0, 0.1 * np.std(node_features), size = node_features.shape)
        node_features =  node_features + np.absolute(noise) # type: ignore
        hics = self.hics[idx] # type: ignore
        abs = self.abs[idx]# type: ignore
        
        # Normalize abs
        max_abs = np.absolute(abs).max()
        abs = abs/max_abs
        
        
        tads = self.tads[idx]
        
        return {
                'node_features': torch.from_numpy(node_features).float(), 
                'hic_targets': torch.from_numpy(hics).float(),
                'ab_targets' : torch.from_numpy(abs).float(), 
                'tad_targets': torch.from_numpy(tads).float(),
        }
        
    def __len__(self) -> int:
        return self.node_features.shape[0]


    

class scGrapHiC(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.PARAMETERS = PARAMETERS
        self.encoder = Encoder(PARAMETERS)
        self.decoder = HiCDecoder(PARAMETERS)
        self.loss_func = torch.nn.MSELoss()
        
        
    def training_step(self, batch):
        x = batch['node_features']
        targets = batch['hic_targets']
        
        targets = targets.view(targets.shape[0], 1, targets.shape[1], targets.shape[2])
        
        x = self.encoder(x)
        x = self.decoder(x, x)
        
        return self.loss(x, targets)
    
    def validation_step(self, batch):
        
        x = batch['node_features']
        print(x.shape)
        
        x = self.encoder(x)
        x = self.decoder(x, x)
        
        targets = batch['hic_targets']
        
        targets = targets.view(targets.shape[0], 1, targets.shape[1], targets.shape[2])
        
        loss = self.loss(x*3, targets*3)
        
        for i in range(x.shape[0]):
            visualize_generated_hic_contact_matrix(
                x[i, 0, :, :].detach().to('cpu').numpy(), 
                targets[i, 0, :, :].detach().to('cpu').numpy(), 
                'visualizations/{}_generated_hic_map.png'.format(i)
            )
        
        return loss


    def predict_step(self, data):
        return 


    def loss(self, generated, targets):
        loss = self.loss_func(generated*3, targets*3)
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer
    


class scGrapHiC_AB(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.PARAMETERS = PARAMETERS
        self.encoder = Encoder(PARAMETERS)
        self.decoder = ABDecoder(PARAMETERS)
        self.loss_func = torch.nn.MSELoss()
        
        
    def training_step(self, batch):
        x = batch['node_features']
        targets = batch['ab_targets']
        
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return self.loss(x, targets)
    
    def validation_step(self, batch):
        
        x = batch['node_features']
        x = self.encoder(x)
        x = self.decoder(x)
        
        print(x.shape)
        targets = batch['ab_targets']
        targets = targets.view(targets.shape[0], targets.shape[1], x.shape[2])
        print(targets.shape)
        
        
        loss = self.loss(x, targets)
        
        for i in range(x.shape[0]):
            visualize_generated_tracks(
                x[i, :, :].detach().to('cpu').numpy(), 
                targets[i, :, :].detach().to('cpu').numpy(), 
                'visualizations/{}_generated_ab_track.png'.format(i)
            )
            visualize_hic_contact_matrix(
                batch['hic_targets'][i, :, :].detach().to('cpu').numpy(), 
                'visualizations/{}_target_hic.png'.format(i)
            )
        
        
        return loss


    def predict_step(self, data):
        return 


    def loss(self, generated, targets):
        loss = self.loss_func(generated*5, targets*5)
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer
    
    
    



class scGrapHiC_TAD(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.PARAMETERS = PARAMETERS
        self.encoder = Encoder(PARAMETERS)
        self.decoder = TADDecoder(PARAMETERS)
        self.loss_func = torch.nn.MSELoss()
        
        
    def training_step(self, batch):
        x = batch['node_features']
        targets = batch['tad_targets']
        
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return self.loss(x, targets)
    
    def validation_step(self, batch):
        
        x = batch['node_features']
        x = self.encoder(x)
        x = self.decoder(x)
        
        print(x.shape)
        targets = batch['tad_targets']
        targets = targets.view(targets.shape[0], targets.shape[1], x.shape[2])
        print(targets.shape)
        
        
        loss = self.loss(x, targets)
        
        for i in range(x.shape[0]):
            visualize_generated_tracks(
                x[i, :, :].detach().to('cpu').numpy(), 
                targets[i, :, :].detach().to('cpu').numpy(), 
                'visualizations/{}_generated_tad_track.png'.format(i)
            )
            visualize_hic_contact_matrix(
                batch['hic_targets'][i, :, :].detach().to('cpu').numpy(), 
                'visualizations/{}_target_hic.png'.format(i)
            )
        
        
        return loss


    def predict_step(self, data):
        return 


    def loss(self, generated, targets):
        loss = self.loss_func(generated*5, targets*5)
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer