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

## Downloading the supplementary datasets and model weights
We have uploaded our model weights and supplementary datasets, such as the CTCF and CpG scores, Gene annotation files, and chrom sizes in this [data](https://drive.google.com/drive/folders/1Bo7sq2TlgVZRU6c6JB4MFAT2LZSz0SEm?usp=sharing) folder. Please download this folder before you proceed with the rest of the installation. While we train and test scGrapHiC on mouse datasets only, we provide these supplementary files for Human (hg38) and Fly (dm6) to aid users who work with these model organisms. We also provide training, testing, and held-out cell-type dataloaders in the processed directory to make it easy to replicate the results in our manuscript. 

We have not uploaded Human and Fly stem cell bulks Hi-C files in this data repository because Hi-C files typically are in order of 10s of GBs. Refer to section XXX later in the README, where we provide links to both Human and Fly bulk Hi-C files. In the same section, we refer interested readers to repositories containing a wide selection of tissue-specific bulk Hi-C measurements. 

## Setting up Paths
Before we run the pipeline, we need to define four static paths in the src/globals.py file: 
- DATA: This path should point to the location where you downloaded. 
- RAW_DATA: This path stores all the raw dataset files. Set this path to point to the 'raw' sub-directory in the folder you downloaded in the previous step. Note: If you want to retrain from scratch, please ensure that the storage directory that stores this path has around 100 GB of free space available. We have provided scripts that autonomously download the HiRES and mouse bulk Hi-C datasets. 
- PREPROCESSED_DATA: This path will contain the pseudo-bulked scRNA-seq and scHi-C datasets. Set it to point to the 'preprocessed' subdirectory in the folder you downloaded in the previous step.
- PROCESSED_DATA: This path will include the dataloader files we use to train and test our model. Set it to point to the 'processed' subdirectory in the folder you downloaded in the previous step.
=======
Before running the pipeline, please change the following variables in src/globals.py file: 
- RAW_DATA: This path stores all the raw dataset files. Please ensure that the storage directory that stores this path has around 100 GB free space available. These raw datasets include HiRES scRNA-seq and scHi-C coassayed dataset, bulk Hi-C datasets, CTCF and CpG scores.  
- PREPROCESSED_DATA: This path will contain the pseudo-bulked scRNA-seq and scHi-C datasets.
- PROCESSED_DATA: This path will include the dataloader files we use to train and test our model. 
- DATA: Folder contains the path where we store weights and generated results. 
>>>>>>> 6a0b7f87223fb1545c1b74d31d5d3cb81c2888d1

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
  
We have provided weights for all these versions in the [data](https://drive.google.com/drive/folders/1Bo7sq2TlgVZRU6c6JB4MFAT2LZSz0SEm?usp=sharing). 

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



# Inference only

If you are not interested in training scGrapHiC from scratch and are interested in only working with the trained model, we have provided another python file 'inference.py' that facilitates that. 'inference.py' follows very similar routines to 'main.py' with an exception of running the entire downloading, preprocessing and dataloader creation pipeline. It assumes that the dataloaders are already in the PROCESSED_DATA directory. 

To run the inference pipeline with the provided dataloaders and the weights: 

```
python inference.py --experiment scGrapHiC --rna_seq True --ctcf_motif True --cpg_motif True --use_bulk True --positional_encodings True --pos_encodings_dim 16 --bulk_hic mesc --hic_smoothing True
```

This should create a new folder in the results directory with the experiment name scGrapHiC which should contain all the generated data for both the testing and held out cell types. 

## Bulk Hi-C for Humans and Flies
While we have only trained and tested scGrapHiC on Mouse (Mus Musculus) datasets, we believe that scGrapHiC can potentially generalize to scRNA-seq datasets from other model organisms. Below, we provide a table with links to embryonic stem cell bulk Hi-C datasets for Humans and Fly. 

### Embryonic Stem Cell bulk Hi-C for other model organisms
Species      | Cell Line |    Dataset link
-------------|-----------|-------------
Human (hg38) |  mESC     | https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/9681f9b5-335a-4f56-afa1-15b58bbb41e8/4DNFI5IAH9H1.hic
Human (hg38) |  H1-hESC  | https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/bb3307fd-7162-477a-87c5-52f12d03befc/4DNFID162B9J.hic
Fly (dm6)    |     S2    | https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3753419&format=file&file=GSM3753419%5FS2%5FInSitu%5FHS%5F1%5Finter%5F30%2Ehic


Once you have downloaded them place them in the "data/raw/your_species/bulk" folder. 

scGrapHiC relies on the bulk Hi-C data to be in the .hic format, but sometimes the available Hi-C data is in .cool format or is in raw FASTQ reads format. To convert .cool (or .mcool) format to .hic you can rely on the [cool2hic](https://github.com/4dn-dcic/docker-4dn-hic/blob/master/README.md#run-mcool2hicsh) conversion scripts provided by 4DN. To process the raw FASTQ reads you can refer to the [Juicer](https://github.com/aidenlab/juicer) pipeline that generates the .hic files from the raw FASTQ reads. 

### Where to acquire tissue specific bulk Hi-C measurements?
As part of our evaluations we have shown that scGrapHiC can adapt to tissue specific bulk Hi-C reads to boost its predictive capacity significantly. Tissue Hi-C measurements are typically closed source and are usually challenging to acquire. However, there are a few data repositories shown in the table below an interested user can browse to find a bulk Hi-C measurement relevant for their analysis. 

Data Repository | Link
-------------   | -------------
Geo Accession   | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
4DN             | https://data.4dnucleome.org/
ENOCDE          | https://www.encodeproject.org/matrix/?type=Experiment&control_type!=*&status=released&perturbed=false

Please feel free to reach out if you need assitance in finding relevant datasets! 

## For other model organisms
We need a total of five supplementary files for scGrapHiC to work for other model organisms. We will go over them one by one explain how to acquire or generate them. 

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

Running this script with save a file with name 'ctcf.bed' that our pre-processing pipeline utilizes to generate both positive and negative strand CTCF motif score node features. Note: you might have to explore other data repositories besides JASPAR to acquire specific CTCF motif scores. 

### CpG

We rely on [pycoMeth](https://a-slide.github.io/pycoMeth) toolkit to generate CG frequency scores for CpG island across the entire genome that we use an auxiliary node feature. pycoMeth requires the reference genome file to generate a genome wide CG frequency scores. We acquire the reference genomes from the [UCSC Genome browser](https://genome.ucsc.edu/cgi-bin/hgGateway?hgsid=2016303202_Qa6m064otsaJjzaDpZnm3ifnqD0s). 

Once you have downloaded this reference file for your species of interest and also have installed the pycoMeth toolkil, just run the following command to generate the genome wide CG frequency scores file: 

```
pycoMeth CGI_Finder -f path/to/your/uncompressed_reference/file.fa -b cpg.bed -t cpg.tsv --progress
```

scGrapHiC takes in the cpg.tsv file to create additional node features. 

### Bulk Hi-C

Best source to acquire bulk Hi-C measurements are the ENCODE and the 4DN portals. However, we would also recommend the users to explore specie specific databanks such as [Flybase](https://flybase.org/) for Drosophilla family species. 

### Gene Annotation files
We need the gene annotation files to reverse map the observed expression back onto their genomic loci. We acquire our gene annotation files from the [GENCODE](https://www.gencodegenes.org/) portal. There are other repositories such as [UCSC browser](https://genome.ucsc.edu/cgi-bin/hgGateway?hgsid=2040118178_Ie1H0irVCicwcBQFXWfpDkARAAhZ). 


### Chrom sizes
You can use this script provided by the [UCSC Genome Browser repository](https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/fetchChromSizes) to download chromsizes file for any species by running the command to fetch chromsizes for humans: 

```
fetchChromSizes hg38 > hg38.chrom.sizes
```

## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/rsinghlab/scGrapHiC/issues)
