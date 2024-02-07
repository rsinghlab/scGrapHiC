# scGrapHiC: Deep learning-based graph deconvolution for Hi-C using single cell gene expression

Single-cell Hi-C (scHi-C) protocol helps identify cell-type-specific chromatin interactions and sheds light on cell differentiation and disease progression. Despite providing crucial insights, scHi-C data is often underutilized due the high cost and the complexity of the experimental protocol. We present a deep learning framework, scGrapHiC, that predicts pseudo-bulk scHi-C contact maps using pseudo-bulk scRNA-seq data. Specifically, scGrapHiC performs graph deconvolution to extract genome-wide single-cell interactions from a bulk Hi-C contact map using scRNA-seq as a guiding signal. Our evaluations show that scGrapHiC, trained on 7 cell-type co-assay datasets, outperforms typical sequence encoder approaches. For example, scGrapHiC achieves a substantial improvement of $23.2\%$ in recovering cell-type-specific Topologically Associating Domains over the baselines. It also generalizes to unseen embryo and brain tissue samples. scGrapHiC is a novel method to generate cell-type-specific scHi-C contact maps using widely available genomic signals that enables the study of cell-type-specific chromatin interactions.

[*Preprint*]() is available.

![scGrapHiC model overview](https://github.com/rsinghlab/scGrapHiC/blob/main/scgraphic_arch.png?raw=true)

## Requirements
We tested and implemented the entire pipeline in python 3.9.16. We highly encourage that you setup a virtual environment before you install required packages to ensure there are no conflicting dependencies. You can setup a new virtual environment through: 

```
python -m venv scgraphic
source scgraphic/bin/activate 
```

Then install all the required packages, run the command:
```
pip install -r requirements.txt
```

## Setting up Paths
Before we run the pipeline, we need to define four static paths in the src/globals.py file: 
- RAW_DATA: This path stores all the raw dataset files. Please ensure that the storage directory that stores this path has around 100 GB free space available. These raw datasets include HiRES scRNA-seq and scHi-C coassayed dataset, bulk Hi-C datasets, CTCF and CpG scores.  
- PREPROCESSED_DATA: This path will contain the pseudo-bulked scRNA-seq and scHi-C datasets.
- PROCESSED_DATA: This path will include the dataloader files we use to train and test our model. 
- DATA: Folder contains the path where we store weights and generated results. 

## Data

In the main.py we have implemented scripts that autonomously downloads all the required datasets into their appropriate directories. The file that implements these functions is src/download_datasets.py. The remote paths are all statically defined and point to 4DN, ENCODE and GEO Accession portals. Please feel free to reach out to us if you have any difficulity in downloading the required datasets through these functions. The main.py file contains these functions to download the datasets:

```
create_directory_structure() # This function also downloads the bulk Hi-C datasets and metadata files
download_hires_schic_datasets()
download_hires_scrnaseq_datasets()
```

Then in the main.py file, we preprocess these datasets through: 

```
# This function parsses the HIRES dataset and prepares it for pseudo-bulking
preprocess_hires_datasets(PARAMETERS)

# These functions then create the pseudo-bulked datasets from the HiRES datasets
create_pseudobulk_files(HIRES_BRAIN_METADATA_FILE)
create_pseudobulk_files(HIRES_EMBRYO_METADATA_FILE)
```

The hyper-parameters for dataset pre-processing are defined in the PARAMETERS object, that we construct from the command line arguments. Run command to print all the hyper-parameters and their default values:
```
python main.py --help
```

Finally, we then we run functions to create the dataloader objects to train our model. 

```
exclusion_set = ['EX15', 'brain']
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'train', PARAMETERS['experiment'])
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'valid', PARAMETERS['experiment'])
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'test', PARAMETERS['experiment'])

exclusion_set = ['E70', 'E75', 'E80', 'E85', 'E95', 'EX05']
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'ood', PARAMETERS['experiment'])

train_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_train.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)
valid_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_valid.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)
test_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_test.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)

ood_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_ood.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)
```
We have defined an exclusion set list that allows us to control which tissue and cell samples to include in our dataloader. This provides us with a high-level abstraction to easily control the contents of different dataloaders we construct. For instance, we have defined an OOD (out of distribution) dataset which excludes all embryo tissue stages except EX15 and brain. We can then evaluate the performance of our model on out-of-distribution datasets that our model doesnt see during training or testing phases. 

## 

