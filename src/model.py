
from importlib import metadata
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch

import numpy as np
import torch.optim as optim
import lightning.pytorch as pl

from src.models.Encoder import ChIPSeqProcessor, GraphEncoder
from src.models.HiCDecoder import HiCDecoder
from src.models.UNet import UnetDecoder

from src.utils import *
from src.graph_pe import graph_pe
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import dense_to_sparse
from src.normalizations import normalizations, smooth_adjacency_matrix
from src.visualizations import log_results, visualize_generated_hic_contact_matrix
from src.evaluations import run_evals




class GenomicDataset(torch.utils.data.Dataset):
    def __init__(self, path, PARAMETERS):
        super().__init__()
        data = np.load(path, allow_pickle=True, mmap_mode='r')
        self.node_features = data['node_features']
        self.targets = data['targets']
        self.pes = data['pes']
        self.bulk_hics = data['bulk_hics']
        self.indexes = data['indexes']
        self.metadatas = data['metadatas']
        self.PARAMETERS = PARAMETERS
        
    def __getitem__(self, idx: int):
        node_features = self.node_features[idx]# type: ignore
        positional_encodings = self.pes[idx]
        
        # 0 - 1 Normalize the node features
        scaler = MinMaxScaler()
        model = scaler.fit(node_features)
        node_features = model.transform(node_features)

        model = scaler.fit(positional_encodings)
        positional_encodings = model.transform(positional_encodings)
        
        targets = self.targets[idx][0, :, :]
        bulk_hics = self.bulk_hics[idx][0, :, :]
         
        indexes = self.indexes[idx]
        metadatas =  self.metadatas[idx]
        
        return {
            'node_features': torch.from_numpy(node_features).float(),
            'positional_encodings': torch.from_numpy(positional_encodings).float(), 
            'targets' : torch.from_numpy(targets).float(), 
            'bulk_hics': torch.from_numpy(bulk_hics).float(),
            'indexes': torch.from_numpy(indexes),
            'metadatas': torch.from_numpy(metadatas),
        }
       
    def __len__(self) -> int:
        return self.node_features.shape[0]


def create_graph_data_object(x, pe, matrix):    
    x = get_node_features(x)
    pe = get_node_features(pe)
    edge_indexes, edge_attrs = dense_to_sparse(matrix)
    return Data(
        x=x,
        pe=pe,
        edge_attr=edge_attrs, 
        edge_index=edge_indexes,
    )

def create_graph_batch(xs, pes, matrices, batch_size):
    graphs = []
    for i in range(matrices.shape[0]):
        graphs.append(
            create_graph_data_object(
                xs[i, :, :],
                pes[i, :, :],
                matrices[i, :, :],
            )
        )
    return next(iter(DataLoader(graphs, batch_size=batch_size, shuffle=False)))


class scGrapHiC(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.PARAMETERS = PARAMETERS
        
        if self.PARAMETERS['ctcf_motif']:
            self.PARAMETERS['node_features'] += 2
        if self.PARAMETERS['cpg_motif']:
            self.PARAMETERS['node_features'] += 1
        
        
        self.chipseq_processor = ChIPSeqProcessor(PARAMETERS)
        self.graph_encoder = GraphEncoder(PARAMETERS)
        self.decoder = HiCDecoder(PARAMETERS)
        self.loss_func = torch.nn.MSELoss()
        
        self.save_hyperparameters()
        
    
    def transform(self, nf, pe, bulk):
        if not self.PARAMETERS['rna_seq']:
            nf = torch.ones_like(nf)
        
        x = self.chipseq_processor(nf, pe)
        graph_batch = create_graph_batch(x, pe, bulk, batch_size=x.shape[0])
        
        graph_latents = self.graph_encoder(
            graph_batch.x,
            graph_batch.pe,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch 
        )
        
        output = self.decoder(graph_latents, graph_latents)
        
        return output
      
    def training_step(self, batch):
        nf = batch['node_features']
        pe = batch['positional_encodings']
        bulk = batch['bulk_hics']
        targets = batch['targets']
        
        output = self.transform(nf, pe, bulk)
                
        targets = targets.view(targets.shape[0], 1, targets.shape[1], targets.shape[2])
        loss = self.loss(output, targets)
        self.log("train/loss", loss)
        print('training/loss', loss)
        
        return loss
    
    
    def validation_step(self, batch):
        nf = batch['node_features']
        pe = batch['positional_encodings']
        bulk = batch['bulk_hics']
        targets = batch['targets']
        
        
        output = self.transform(nf, pe, bulk)
        
        targets = targets.view(targets.shape[0], 1, targets.shape[1], targets.shape[2])
        
        scores = run_evals(output, targets)
        loss = self.loss(output, targets)
        
        self.log("valid/loss", loss)
        
        self.log("valid/MSE", np.mean(scores['MSE']))
        self.log("valid/SSIM", np.mean(scores['SSIM']))
        self.log("valid/GD", np.mean(scores['GD']))
        self.log("valid/SCC", np.mean(scores['SCC']))

        return loss


    def test_step(self, batch, batch_idx):
        nf = batch['node_features']
        pe = batch['positional_encodings']
        bulk = batch['bulk_hics']
        targets = batch['targets']
        indexes = batch['indexes']
        metadatas = batch['metadatas']
        
        
        output = self.transform(nf, pe, bulk)
        
        targets = targets.view(targets.shape[0], 1, targets.shape[1], targets.shape[2])
        
        scores = run_evals(output, targets)
        
        for i in range(output.shape[0]):
            log_results(
                output[i, 0, :, :],
                targets[i, 0, :, :],
                [scores['MSE'][i], scores['SSIM'][i], scores['GD'][i], scores['SCC'][i]],
                indexes[i, :],
                metadatas[i, :],
                self.PARAMETERS
            )

    
    def loss(self, generated, targets):
        loss = self.loss_func(
            generated*self.PARAMETERS['loss_scale'], 
            targets*self.PARAMETERS['loss_scale']
        )
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer