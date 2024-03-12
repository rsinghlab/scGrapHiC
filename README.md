# scGrapHiC: Deep learning-based graph deconvolution for Hi-C using single cell gene expression

Single-cell Hi-C (scHi-C) protocol helps identify cell-type-specific chromatin interactions and sheds light on cell differentiation and disease progression. Despite providing crucial insights, scHi-C data is often underutilized due the high cost and the complexity of the experimental protocol. We present a deep learning framework, scGrapHiC, that predicts pseudo-bulk scHi-C contact maps using pseudo-bulk scRNA-seq data. Specifically, scGrapHiC performs graph deconvolution to extract genome-wide single-cell interactions from a bulk Hi-C contact map using scRNA-seq as a guiding signal. Our evaluations show that scGrapHiC, trained on 7 cell-type co-assay datasets, outperforms typical sequence encoder approaches. For example, scGrapHiC achieves a substantial improvement of $23.2\%$ in recovering cell-type-specific Topologically Associating Domains over the baselines. It also generalizes to unseen embryo and brain tissue samples. scGrapHiC is a novel method to generate cell-type-specific scHi-C contact maps using widely available genomic signals that enables the study of cell-type-specific chromatin interactions.

[*Preprint*](https://www.biorxiv.org/content/10.1101/2024.02.07.579342v1) is available.

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

train_data_loader =  torch.utils.data.DataLoader(train_dataset, PARAMETERS['batch_size'], shuffle=True)
validation_data_loader =  torch.utils.data.DataLoader(valid_dataset, PARAMETERS['batch_size'], shuffle=False)
test_data_loader = torch.utils.data.DataLoader(test_dataset, PARAMETERS['batch_size'], shuffle=False)
ood_data_loader = torch.utils.data.DataLoader(ood_dataset, PARAMETERS['batch_size'], shuffle=False)

```
We have defined an exclusion set list that allows us to control which tissue and cell samples to include in our dataloader. This provides us with a high-level abstraction to easily control the contents of different dataloaders we construct. For instance, we have defined an OOD (out of distribution) dataset which excludes all embryo tissue stages except EX15 and brain. We can then evaluate the performance of our model on out-of-distribution datasets that our model doesnt see during training or testing phases. 

## Model and Baselines
The model is implemented with Pytorch and Pytorch Geometric. We have implemented scGrapHiC in such a way that by controlling the command line parameter we can convert it into different baseline implementations. For example:

- Bulk Only: We set command line parameters as --experiment bulk_only --rna_seq False --use_bulk True --positional_encodings False. Setting rna_seq False automatically ignores CTCF and CpG. 
- scRNA-seq Only: --experiment rna_seq_only --rna_seq True, this does not include CTCF and CpG. 
- scRNA-seq + CTCF: --experiment rna_seq_ctcf --rna_seq True --ctcf_motif True 
- scRNA-seq + CTCF + CpG: --experiment rna_seq_ctcf_cpg --rna_seq True --ctcf_motif True --cpg_motif True
- scGrapHiC: --experiment scgraphic --rna_seq True --ctcf_motif True --cpg_motif True --use_bulk True --positional_encodings True --pos_encodings_dim 16
  
```
tb_logger = TensorBoardLogger("logs", name=PARAMETERS['experiment'])
checkpoint_callback = ModelCheckpoint(monitor="valid/SCC",  save_top_k=3, mode='max')
scgraphic = scGrapHiC(PARAMETERS)

trainer = pl.Trainer(
    max_epochs=PARAMETERS['epochs'], 
    check_val_every_n_epoch=50, 
    logger=tb_logger,
    deterministic=True,
    callbacks=[checkpoint_callback],
    gradient_clip_val=PARAMETERS['gradient_clip_value'],
)

trainer.fit(scgraphic, train_data_loader, validation_data_loader)
```
We save top-3 versions of scGrapHiC that maximize the performance on Stratum-Adjusted Correlation Coefficient scores as shown through the checkpoint callback code. We also do gradient clipping to ensure stable training and have set static seeds and configured deterministic training to ensure our model provides same performance on same seed across different machines configurations. Our model was trained with a single RTX 3090 GPU. 

We then finally test and evaluate our model through the functions:

```
trainer.test(scgraphic, test_data_loader) 
trainer.test(scgraphic, ood_data_loader)

evaluate(os.path.join(RESULTS, PARAMETERS['experiment']), PARAMETERS)
```

To excute the full pipeline just run: 
```
python main.py --experiment scgraphic --rna_seq True --ctcf_motif True --cpg_motif True --use_bulk True --positional_encodings True --pos_encodings_dim 16
```

## Setup for other model species
We have only trained and tested scGrapHiC for Mus Musculus (mouse) datasets, and given our data availability constraints, we could not test our model on other model organisms. This section discusses acquiring and pre-processing auxiliary data such as CTCF, CpG, and bulk Hi-C for other model organisms. We show examples of Mus Musculus (mouse), Homo Sapiens (humans), and Drosophila Melanogaster (fruit fly). 

### CTCF
To generate cell agnostic CTCF motif scores we use the R package [CTCF](https://github.com/dozmorovlab/CTCF). Once you have installed the CTCF package run the code below tailored to your species of choice to generate the .bed files: 

```
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(version = "3.18")


BiocManager::install("AnnotationHub", update = FALSE) 
BiocManager::install("GenomicRanges", update = FALSE)
BiocManager::install("plyranges", update = FALSE)


suppressMessages(library(AnnotationHub))
suppressMessages(library(GenomicRanges))
suppressMessages(library(plyranges))

ah <- AnnotationHub()
query_data <- subset(ah, preparerclass == "CTCF")

# subset(query_data, species == "Homo sapiens" & genome == "hg38" &  dataprovider == "JASPAR 2022") # for human hg38
# subset(query_data, species == "Homo sapiens" & genome == "hg19" &  dataprovider == "JASPAR 2022")  # for human hg19
subset(query_data, species == "Mus musculus" & genome == "mm10" & dataprovider == "JASPAR 2022") # for mouse mm10

# See the output of the query data and get the key associated with the JASPAR2022_CORE_vertebrates_non_redundant_v2.RData to get CTCF motif scores with all 3 PWMs in JASPAR database.
# query_data --> uncomment this to find out the key 



# CTCF_all <- query_data[["AH104727"]] # For human hg38
# CTCF_all <- query_data[["AH104736"]] # For human hg19
CTCF_all <- query_data[["AH104753"]] # For mouse mm10 


CTCF_all <- CTCF_all %>% keepStandardChromosomes() %>% sort()
CTCF_all <- CTCF_all %>% plyranges::filter(pvalue < 1e-6) # We do not want noisy, cell-type specific motif measurements.

write.table(CTCF_all %>% sort() %>% as.data.frame(), file = "ctcf.bed", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

```

Running this script with save a file with name 'ctcf.bed' that our pre-processing pipeline utilizes to generate both positive and negative strand CTCF motif score node features. 

### CpG
We rely on [pycoMeth](https://a-slide.github.io/pycoMeth) toolkit to generate CG frequency scores for CpG island across the entire genome that we use an auxiliary node feature. pycoMeth requires the reference genome file to generate a genome wide CG frequency scores. We acquire the reference genomes from the [UCSC Genome browser](https://genome.ucsc.edu/cgi-bin/hgGateway?hgsid=2016303202_Qa6m064otsaJjzaDpZnm3ifnqD0s). For ease of access, we provide links to the reference genomes of three popular model species:

- Homo Sapien [(hg38)](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz), [(hg19)](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz) 
- Mus Musculus [(mm10)](https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz)

Once you have downloaded this reference file and also have installed the pycoMeth toolkil, just run the following command to generate the genome wide CG frequency scores file: 

```
pycoMeth CGI_Finder -f path/to/your/uncompressed_reference/file.fa -b cpg.bed -t cpg.tsv --progress
```

### Bulk Hi-C

We acquired our bulk Hi-C files from the 4DN and the ENCODE portal. As shown in our evaluation for most deconvolution tasks, a embryonic stem cell bulk Hi-C measurements is enough. In the table below we provide links to 

Species -- Assembly  | Embryonic Stem Cells (ESC) bulk Hi-C datasets
------------- | -------------
Homo Sapiens -- hg19  | https://data.4dnucleome.org/experiment-set-replicates/4DNESFSCP5L8/#raw-files
Homo Sapiens -- hg38  | https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/bb3307fd-7162-477a-87c5-52f12d03befc/4DNFID162B9J.hic
Mus Musculus -- mm10  | https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/9681f9b5-335a-4f56-afa1-15b58bbb41e8/4DNFI5IAH9H1.hic
--------------------------

Note that for hg19 assembly we only have raw FASTQ files available, and to generate .hic files we have to run the Juicer pipeline with hg19 assembly and parameters. Refer to [Juicer](https://github.com/aidenlab/juicer) for more details. 


For specific tissue specific use cases, we recommend that the users browse the [ENCODE](https://www.encodeproject.org/matrix/?type=Experiment&assay_title=intact+Hi-C&assay_title=in+situ+Hi-C&assay_title=dilution+Hi-C) and [4DN](https://data.4dnucleome.org/) portal to acquire a bulk Hi-C for the tissue they have sequenced using their single-cell RNA-seq protocol. 


In case the Hi-C file is in .mcool or .cool format use the script provided by the 4DN at the [link](https://github.com/4dn-dcic/docker-4dn-hic/blob/master/README.md#run-mcool2hicsh).



### Note for other species
The steps we have provided should work for any arbitrary species as long as they have the supporting data available on these databases. Unfortunately, our only data availability constraint is for CTCF, where the JASPAR database currently only contains motifs for Human and Mouse. We have provided alternative scGrapHiC implementations that rely on bulk Hi-C, CpG, and scRNA-seq alone; interested users are encouraged to try those implementations. In the future, we aim to test how our model performs if we provide a CTCF ChIP-seq measurement instead for species not covered under the JASPAR database. 

Please feel free to reach out to us if you have an problems processing these datasets. 

## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/rsinghlab/scGrapHiC/issues)
