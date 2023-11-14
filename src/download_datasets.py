'''
    This file downloads the required datasets into appropriate folders
'''
import os
from venv import create
import wget

from src.globals import *
from src.utils import create_directory

def create_directory_structure():
    '''
    Create all the necessary folders and download the necessary files
    '''
    # Top level structure
    create_directory(RAW_DATA)
    create_directory(PREPROCESSED_DATA)
    create_directory(PROCESSED_DATA)

    # Internal folders
    create_directory(MOUSE_RAW_DATA_HIRES)
    create_directory(MOUSE_RAW_DATA_SCHIC)
    create_directory(MOUSE_RAW_DATA_SCRNASEQ)
    create_directory(MOUSE_RAW_BULK_DATA)
    create_directory(MOUSE_RAW_DATA_PSEUDO_BULK)
    create_directory(MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ)
    create_directory(MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC)
    
    
    create_directory(MOUSE_PREPROCESSED_DATA_HIRES)
    create_directory(MOUSE_PREPROCESSED_DATA_SCHIC)
    create_directory(MOUSE_PREPROCESSED_DATA_SCRNASEQ)
    create_directory(MOUSE_PREPROCESSED_DATA_BULK)
    
    
    create_directory(MOUSE_PROCESSED_DATA_HIRES)
    
    
    # Download the series matrix file
    if not os.path.exists(HIRES_SERIES_MATRIX_FILE):
        wget.download(
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223917/matrix/GSE223917_series_matrix.txt.gz',
            HIRES_SERIES_MATRIX_FILE
        )
    # Download the mm10 GTF3 file
    if not os.path.exists(MM10_GTF3_FILE_PATH):
        wget.download(
            'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M23/gencode.vM23.annotation.gtf.gz',
            MM10_GTF3_FILE_PATH
        )

    # Download the bulk Hi-C datasets
    if not os.path.exists(MOUSE_PN5_ZYGOTE_BULK_HIC):
        wget.download(
            'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/1181c0c4-afb7-4b6a-9fdc-d868fb2253fc/4DNFI1EYIGOC.hic',
            MOUSE_PN5_ZYGOTE_BULK_HIC
        )
    
    if not os.path.exists(MOUSE_MESC_BULK_HIC):
        wget.download(
            'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/9681f9b5-335a-4f56-afa1-15b58bbb41e8/4DNFI5IAH9H1.hic',
            MOUSE_MESC_BULK_HIC
        )
    
    if not os.path.exists(MOUSE_CEREBRAL_CORETEX_BULK_HIC):
        wget.download(
            'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/268b7d52-9655-474c-9467-8ba31bb2195c/4DNFII3JV8I1.hic',
            MOUSE_CEREBRAL_CORETEX_BULK_HIC
        )
    
    if not os.path.exists(HIRES_BRAIN_METADATA_FILE):
        wget.download(
            'https://0-www-ncbi-nlm-nih-gov.brum.beds.ac.uk/geo/download/?acc=GSE223917&format=file&file=GSE223917%5FHiRES%5Fbrain%5Fmetadata%2Exlsx',
            HIRES_BRAIN_METADATA_FILE
        )
    
    if not os.path.exists(HIRES_EMBRYO_METADATA_FILE):
        wget.download(
            'https://0-www-ncbi-nlm-nih-gov.brum.beds.ac.uk/geo/download/?acc=GSE223917&format=file&file=GSE223917%5FHiRES%5Femb%5Fmetadata%2Exlsx',
            HIRES_EMBRYO_METADATA_FILE
        )
    


def download_hires_schic_datasets():
    '''
        Downloads the scHi-C datasets from the HiRES experiment
    '''
    file = open(HIRES_SERIES_MATRIX_FILE, mode = 'r', encoding = 'utf-8-sig')
    lines = file.readlines()
    file.close()
    urls = ""

    # Parse through text file to get list of urls
    for line in lines:
        if line[:28] == '!Sample_supplementary_file_2':
            urls = line
            print(line[:50])

    # Create list by splitting on whitespace and replacing 'ftp://ftp...' with 'https://ftp...'
    my_list = urls.split()[1:]
    for i, link in enumerate(my_list):
        my_list[i] = 'https' + link[4:-1]


    for link in my_list:
        file_name = link.split('/')[-1].split('_')[-1]
        local_link = os.path.join(MOUSE_RAW_DATA_SCHIC, file_name)        
        if os.path.exists(local_link):
            continue
        wget.download(link, local_link)


def download_hires_scrnaseq_datasets():
    '''
        Downloads the scRNA-Seq datasets from the HiRES experiment
    '''
    brain_scrnaseq_data = os.path.join(
        MOUSE_RAW_DATA_SCRNASEQ, 
        'GSE223917_HiRES_brain.rna.umicount.tsv.gz'
    )
    if not os.path.exists(brain_scrnaseq_data):
        wget.download(
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223917/suppl/GSE223917_HiRES_brain.rna.umicount.tsv.gz', 
            brain_scrnaseq_data
    )

    embryo_scrnaseq_data = os.path.join(
        MOUSE_RAW_DATA_SCRNASEQ, 
        'GSE223917_HiRES_emb.rna.umicount.tsv.gz'
    )

    if not os.path.exists(brain_scrnaseq_data):
        wget.download(
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE223nnn/GSE223917/suppl/GSE223917_HiRES_emb.rna.umicount.tsv.gz', 
            embryo_scrnaseq_data
    )



def psuedo_bulk_divisions():
    
    return 




