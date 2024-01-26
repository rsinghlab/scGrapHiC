import math
import torch
import numpy as np
import torch.nn.functional as F


from scipy import sparse
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    from_scipy_sparse_matrix
)

def is_perfect_square(num):
    if num < 0:
        return False

    # Calculate the square root
    square_root = math.isqrt(num)

    # Check if the square root is equal to its integer value
    return square_root * square_root == num


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom
    
    # Finally put it between 0 and 1 if abs max
    if normalization == 'abs-max':
        EigVecs = (EigVecs + 1)/2.0    
    
    
    return EigVecs


def graph_pe(matrix, encoding_dim, lap_norm='sym', eig_norm='abs-max'):
    """
    Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    encoding_dim += 1
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    

    if len(matrix.shape) == 3:
        matrix  = matrix[0, :, :]
    
    N = matrix.shape[0]

    
    
    sparse_matrix = sparse.csr_matrix(matrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse_matrix)
    
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization=lap_norm, num_nodes=N)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, N)

    evals, evects = np.linalg.eigh(L.toarray())
    
    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:encoding_dim]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)
     # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eig_norm)
    
    if N < encoding_dim:
        EigVecs = F.pad(evects, (0, encoding_dim - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < encoding_dim:
        EigVals = F.pad(evals, (0, encoding_dim - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    
    return EigVecs.cpu().detach().numpy()[:, 1:]

