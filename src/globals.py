'''
    This file contains all the static definitions I plan to use in the project
    
'''
import os

DATA = '/users/gmurtaza/data/gmurtaza/scGrapHiC/'
RAW_DATA = '/users/gmurtaza/data/gmurtaza/scGrapHiC/raw'
PREPROCESSED_DATA = '/users/gmurtaza/data/gmurtaza/scGrapHiC/preprocessed'
PROCESSED_DATA = '/users/gmurtaza/data/gmurtaza/scGrapHiC/processed'

MOUSE_RAW_DATA = os.path.join(RAW_DATA, 'mm10')
MOUSE_PREPROCESSED_DATA = os.path.join(PREPROCESSED_DATA, 'mm10')
MOUSE_PROCESSED_DATA = os.path.join(PROCESSED_DATA, 'mm10')

MOUSE_RAW_DATA_HIRES = os.path.join(MOUSE_RAW_DATA, 'HiRES')
MOUSE_RAW_DATA_SCHIC = os.path.join(MOUSE_RAW_DATA_HIRES, 'scHi-C')
MOUSE_RAW_DATA_SCRNASEQ = os.path.join(MOUSE_RAW_DATA_HIRES, 'scRNA-seq')
MOUSE_RAW_BULK_DATA = os.path.join(MOUSE_RAW_DATA, 'bulk')

MOUSE_RAW_DATA_PSEUDO_BULK = os.path.join(MOUSE_RAW_DATA_HIRES, 'pseudo-bulk')
MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ = os.path.join(MOUSE_RAW_DATA_PSEUDO_BULK, 'scRNA-seq')
MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC = os.path.join(MOUSE_RAW_DATA_PSEUDO_BULK, 'scHi-C')



MOUSE_PREPROCESSED_DATA_HIRES = os.path.join(MOUSE_PREPROCESSED_DATA, 'HiRES')
MOUSE_PREPROCESSED_DATA_SCHIC = os.path.join(MOUSE_PREPROCESSED_DATA_HIRES, 'scHi-C')
MOUSE_PREPROCESSED_DATA_SCRNASEQ = os.path.join(MOUSE_PREPROCESSED_DATA_HIRES, 'scRNA-seq')

MOUSE_PREPROCESSED_DATA_PSEUDO_BULK = os.path.join(MOUSE_PREPROCESSED_DATA_HIRES, 'pseudo-bulk')
MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ = os.path.join(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK, 'scRNA-seq')
MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC = os.path.join(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK, 'scHi-C')


MOUSE_PREPROCESSED_DATA_BULK = os.path.join(MOUSE_PREPROCESSED_DATA, 'bulk')

MOUSE_PROCESSED_DATA_HIRES = os.path.join(MOUSE_PROCESSED_DATA, 'HiRES')



HIRES_SERIES_MATRIX_FILE = os.path.join(MOUSE_RAW_DATA_HIRES,'GSE223917_series_matrix.txt')
HIRES_BRAIN_METADATA_FILE = os.path.join(MOUSE_RAW_DATA_HIRES, 'metadata', 'brain_metadata.xlsx')
HIRES_EMBRYO_METADATA_FILE = os.path.join(MOUSE_RAW_DATA_HIRES, 'metadata', 'embryo_metadata.xlsx')


MM10_GTF3_FILE_PATH = os.path.join(MOUSE_RAW_DATA, 'gencode.vM23.annotation.gff3.gz')
MM10_CPG_PATH = os.path.join(MOUSE_RAW_DATA, 'cpg')


# BULK HIC DATASETS
MOUSE_PN5_ZYGOTE_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'pn5_zygote.hic')
MOUSE_MESC_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'mesc.hic')
MOUSE_CEREBRAL_CORETEX_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'cerebral_cortex.hic')