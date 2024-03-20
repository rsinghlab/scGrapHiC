'''
    This file contains the functions to preprocess the HiRES dataset
'''
import os
import time

import cooler
import hicstraw

import scanpy as sc
import pandas as pd

from src.utils import *
from src.globals import *
from anndata import AnnData
from multiprocessing import Process
from scipy.sparse import csr_matrix
from src.compartments import ABcompartment_from_mat
from src.TADs import insulation_score
from src.visualizations import visualize_hic_contact_matrix, visualize_scnrna_seq_tracks



def get_gene_name(attributes):
    '''
        Extracts the gene name from the GTF3 file attributes dictionary
    '''
    attributes = attributes.split(';')
    attribute = list(filter(lambda x: 'gene_name' in x, attributes))[0]
    attribute = attribute.split('=')[-1]
    return attribute



def process_gtf3_file(gtf3_filepath, output_path):
    '''
        Process the GTF3 file to get the geneome coordinate maping from the gene names, 
        its a supporting file required for processing RNA-seq UMI cell-by-gene matrices
    '''

    data = pd.read_csv(
        gtf3_filepath, header = None,
        comment ='#', sep ='\t', 
        names=[
            'seqid', 
            'source', 'type',
            'start', 'end', 
            'score', 'strand', 'phase', 
            'attributes'
        ]
    )
    # We only need the genes annotations
    data = data.loc[data['type'] == 'gene']
    # We only need the gene name
    data['attributes'] = data['attributes'].map(get_gene_name)

    data = data.drop(columns=['source', 'type', 'score', 'phase'])
    data = data.rename(columns={'seqid': 'chr', 'attributes': 'gene_name'})
    
    print('Created the Gene Coordinate file, saving it at: {}'.format(output_path))
    data.to_csv(output_path, index=False)


def read_cell_by_gene_matrix(path):
    format = path.split('/')[-1].split('.')[-2]
    
    if format == 'csv':
        sep = ','
    elif format == 'tsv':
        sep = '\t'
    else:
        sep = None
    
    cell_by_gene_data = pd.read_csv(
        path, sep=sep,
        comment='#'
    )
    
    return cell_by_gene_data

def normalize_cell_by_gene_matrix(cell_by_gene_matrix):
    X = cell_by_gene_matrix.iloc[1:, 1:].to_numpy()
    genes = cell_by_gene_matrix.iloc[1:, 0:1].to_numpy().reshape(-1)
    cells = cell_by_gene_matrix.columns[1:].to_numpy()
    
    adata = AnnData(X.T)
    X_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
    X_norm = X_norm.T
    cell_by_gene_matrix = pd.DataFrame(data=X_norm, index=genes, columns=cells)
    cell_by_gene_matrix = cell_by_gene_matrix.reset_index().rename(columns={'index':'gene'})	
    
    return cell_by_gene_matrix

def convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_matrix, gene_coordinate_file):
    gene_coordinates = pd.read_csv(gene_coordinate_file)
    
    # Left join on both tables and we drop the gene_names and have NaNs at genes that were filtered, we replace them with 0s. 
    merged_tables = pd.merge(gene_coordinates, cell_by_gene_matrix, left_on='gene_name', right_on='gene', how='left').drop('gene_name', axis=1)

    return merged_tables


def create_coordinate_matrix_file(scrna_seq_file, gene_cooridate_file_path, output_folder, PARAMETERS):
    scrna_seq_file_name = scrna_seq_file.split('/')[-1].split('.')[0]
    output_intermediate_file = os.path.join(output_folder, scrna_seq_file_name + '.csv')
    
    if os.path.exists(output_intermediate_file):
        print('Coordinate matrix file {} already exists'.format(output_intermediate_file))
        return output_intermediate_file
    
    # Step 1: read the file into a pandas dataframe
    cell_by_gene_data = read_cell_by_gene_matrix(scrna_seq_file)

    # Step 1.5: normalize the cell-by-gene matrix file 
    if PARAMETERS['normalize_umi']:
        cell_by_gene_data = normalize_cell_by_gene_matrix(cell_by_gene_data)
    
    # Step 2: convert it into a gene coordinate format
    coordinate_matrix = convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_data, gene_cooridate_file_path)
    
    # Checkpoint here, and store the coordinate matrix file
    print('Saving coordinate matrix file {}'.format(output_intermediate_file))
    coordinate_matrix.to_csv(output_intermediate_file)
    
    
    return output_intermediate_file
    
    
    
    
def merge_chr_coordinates(coordinates):
    # Aggregate the UMI reads based on the starting and ending coordinate
    coordinates = coordinates.groupby(['start', 'end']).sum()
    # Drop some useless rows
    coordinates = coordinates.drop(['chr', 'strand', 'gene'], axis=1)
    coordinates = coordinates.reset_index()

    return coordinates


def normalize_genomic_track(reads, PARAMETERS):
    sum_reads = np.sum(reads)
    reads = np.divide(reads, sum_reads)
    reads = reads * PARAMETERS['library_size']
    reads = np.log1p(reads)
    reads = reads/np.max(reads)
    
    return reads
    

def create_genomic_track(starts, ends, reads, chrsize, PARAMETERS):
    size = chrsize // PARAMETERS['resolution'] + 1
    
    track = np.zeros(size)
    
    for i in range(starts.shape[0]):
        track[starts[i]:ends[i] + 1] += reads[i]/((ends[i] + 1) - starts[i])
    
    if PARAMETERS['normalize_track']:
        track = normalize_genomic_track(track, PARAMETERS)
    
    return track



def create_genomic_track_file(preprocessed_coordinate_matrix, PARAMETERS, output_path, pseudobulk=False):
    chrom_sizes = read_chromsizes_file(os.path.join(MOUSE_RAW_DATA, 'chrom.sizes'))    
    coordinate_matrix = pd.read_csv(preprocessed_coordinate_matrix)
    coordinate_matrix = coordinate_matrix.loc[:, ~coordinate_matrix.columns.str.contains('^Unnamed')]
    
    # Step 3: proces this coordinate matrix with pandas operations
    # Step 3.1: Replace NaNs with zeros
    coordinate_matrix.dropna(subset=['gene'], inplace=True)
    coordinate_matrix = coordinate_matrix.fillna(0)
    
    # Step 3.2: Convert coordinate scale in accordance with the resolution
    coordinate_matrix['start'] = coordinate_matrix['start'].copy().floordiv(PARAMETERS['resolution'])
    coordinate_matrix['end'] = coordinate_matrix['end'].copy().floordiv(PARAMETERS['resolution'])
    
    # Step 3.3: divide the positive and negative strands into two dataframes
    positive_strand_coordinates = coordinate_matrix.loc[coordinate_matrix['strand'] == '+']
    negative_strand_coordinates = coordinate_matrix.loc[coordinate_matrix['strand'] == '-']
    
    # Step 3.4: All cell types
    cells = list(coordinate_matrix.columns[5:])
    
    # Step 4: for each chromosome extract the track for each cell
    for chromosome, size in chrom_sizes.items():
        # step 4.1: extract reads only specific to a chromosome
        chr_positive_strand_coordinates = positive_strand_coordinates.loc[positive_strand_coordinates['chr'] == chromosome]
        chr_negative_strand_coordinates = negative_strand_coordinates.loc[negative_strand_coordinates['chr'] == chromosome]
        
        # step 4.2: merge the coordinates that overlap
        chr_positive_strand_coordinates = merge_chr_coordinates(chr_positive_strand_coordinates)
        chr_negative_strand_coordinates = merge_chr_coordinates(chr_negative_strand_coordinates)
        
        #step 4.3: for each cell extract the tracks
        for cell in cells:
            if pseudobulk:
                # For pseudo-bulking we have to store the other paramters of pseudobulking
                stage, tissue, cell_type, num_cells = get_file_name_parameters(preprocessed_coordinate_matrix)
                folder = '_'.join([stage, tissue, cell_type, 'n{}'.format(num_cells), 'scrnaseq']) if stage else  '_'.join([tissue, cell_type, 'n{}'.format(num_cells, 'scrnaseq')])
                output_folder = os.path.join(output_path, folder)
            else:
                output_folder = os.path.join(output_path, cell)
            
            create_directory(output_folder)
            
            output_file = os.path.join(output_folder, '{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
            
            #step 4.4: extract both postive and negative tracks
            positive_track = create_genomic_track(
                chr_positive_strand_coordinates['start'].to_numpy(), 
                chr_positive_strand_coordinates['end'].to_numpy(), 
                chr_positive_strand_coordinates[cell].to_numpy(), 
                size,
                PARAMETERS
            )
            
            negative_track = create_genomic_track(
                chr_negative_strand_coordinates['start'].to_numpy(), 
                chr_negative_strand_coordinates['end'].to_numpy(), 
                chr_negative_strand_coordinates[cell].to_numpy(), 
                size,
                PARAMETERS
            )
            
            combined_tracks = np.stack((positive_track, negative_track))
            
            # save the tracks
            print('Saving: {}'.format(output_file))
            
            np.save(output_file, combined_tracks)



def parse_hires_scrnaseq_datasets(input_path, PARAMETERS, output_path, pseudobulk=False):
    '''
        This function parses the scRNA-seq datasets from cell-by-gene to geneome coordinate tracks
    '''
    
    #Step 0: Setting up auxiliary files 
    #Since its a mouse dataset aligned on mm10 assembly, we first create the mm10 gene_coordinate track
    gene_cooridate_file_path = os.path.join(PREPROCESSED_DATA, 'gene_coordinates.csv')
    if not os.path.exists(gene_cooridate_file_path):
        process_gtf3_file(
            MM10_GTF3_FILE_PATH,
            gene_cooridate_file_path
        )

    
    
    
    scrna_seq_files = list(map(lambda x: os.path.join(input_path, x), os.listdir(input_path)))
    
    preprocessed_coordinate_matrices = []
    # For all the cell-by-gene UMI matrix files we do
    for scrna_seq_file in scrna_seq_files:
        output_intermediate_file = create_coordinate_matrix_file(
            scrna_seq_file,
            gene_cooridate_file_path,
            output_path,
            PARAMETERS
        )
        preprocessed_coordinate_matrices.append(output_intermediate_file)
    
    
    for preprocessed_coordinate_matrix in preprocessed_coordinate_matrices:
        create_genomic_track_file(
            preprocessed_coordinate_matrix, 
            PARAMETERS, 
            output_path,
            pseudobulk
        )
    
        
                




########################################################################################################################################################################


def read_pairix_file(path):
    '''
        This function reads a pairix file format (.pairs) and returns 
        a dictionary of numpy arrays
        @params: <string> - path, path to the file 
        @returns: <pd.DataFrame> - pandas dataframe 
    '''
    if os.path.exists(path):
        data = pd.read_csv(
            path, header = None,
            comment ='#', sep ='\t', 
            names=[
                'readID', 
                'chr1', 'pos1',
                'chr2', 'pos2', 
                'strand1', 'strand2', 
                'phase0', 'phase1'
            ],
            dtype={
                "readID": str, 
                "chr1": str, "pos1": int,
                "chr2": str, "pos1": int,
                'strand1': str, 'strand2': str, 
                'phase0': str , 'phase1': str   
            }

        )
        data['pos1'] = pd.to_numeric(data["pos1"])
        data['pos2'] = pd.to_numeric(data["pos2"])

        return data    


def convert_pairs_to_pixels(dataframe, PARAMETERS):
    dataframe.loc[:, 'pos1'] = dataframe['pos1'].copy().floordiv(PARAMETERS['resolution'])
    dataframe.loc[:, 'pos2'] = dataframe['pos2'].copy().floordiv(PARAMETERS['resolution'])
    pixels = dataframe.groupby(['pos1', 'pos2']).size().reset_index(name='counts')
    pixels = pixels.rename(columns={'pos1': 'bin1_id', 'pos2': 'bin2_id', 'counts': 'count'}) 
    return pixels

  

            
def parse_hires_schic_datasets(input_path, PARAMETERS, output_path):
    chrom_sizes = read_chromsizes_file(os.path.join(MOUSE_RAW_DATA, 'chrom.sizes'))
    schic_files = list(map(lambda x: os.path.join(input_path, x), os.listdir(input_path)))            
    
    
    for schic_file in schic_files:
        if '.pairs' in schic_file and os.path.exists(schic_file):
            # Step 0: create the output directory
            cell_name = schic_file.split('/')[-1].split('.')[0]
            output_directory = os.path.join(output_path, cell_name)
            create_directory(output_directory)
            
            # Step 1: parse the .pairs.gz file 
            pairs_data = read_pairix_file(schic_file)
            pairs_data = pairs_data.drop(['readID', 'strand1', 'strand2', 'phase0', 'phase1'], axis=1)
            
            
            for chromosome, size in chrom_sizes.items():
                
                output_cooler_file = os.path.join(output_directory, '{}_{}.cool'.format(chromosome, PARAMETERS['resolution']))
                output_numpy_file = os.path.join(output_directory, '{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
                
                chrom_data = pairs_data.loc[(pairs_data['chr1'] == chromosome) & (pairs_data['chr2'] == chromosome)]
                chrom_pixels = convert_pairs_to_pixels(chrom_data, PARAMETERS)
                bins = chrom_bins(chromosome, size, PARAMETERS['resolution'])

                cooler.create_cooler(output_cooler_file, bins, chrom_pixels,
                                dtypes={"count":"int"}, 
                                assembly="mm10")
                
                # Read and normalize and save as a npy file
                clr = cooler.Cooler(output_cooler_file)
                matrix = clr.matrix(balance=False).fetch(chromosome)

                #  Saving the entire matrix 
                print('Saving: {}'.format(output_numpy_file))
                np.save(output_numpy_file, matrix)
                



############################################################################################################################################################################


# Main Multiprocessing switch for the HiC parser
MULTIPROCESSING=False

def process_chromosome(hic, output, resolution, chromosome, debug):
    '''
        This function handles the bulk of the work of extraction of Chromosome from
        the HiC file and storing it in its dense 2D contact map form
        @params: hic <hicstraw.HiCFile>, HiC file object as returned by the hicstraw utility
        @params: output <os.path>, path where to store the output files
        @params: resolution <int>, resolution to sample the HiC data at
        @params: chromosome <hicstraw.chromosome>, hicstraw chromosome objects that contains its name and misc properties
        @returns: None
    '''
    index = chromosome.index
    length = chromosome.length
    name = chromosome.name
    
    name = name.split('chr')[-1]

    output_path = os.path.join(
        output, 
        'chr{}_{}.npz'.format(name, resolution)
    )
    if os.path.exists(output_path):
        print('Already parsed!')
        return
        
    if name in ['Y','MT', 'M']:
        return 

    if os.path.exists(output_path):
        if debug: print('Already parsed!')
        return

    if debug: print('Starting parsing Chromosome {}'.format(name))

    try:
        chromosome_matrix = hic.getMatrixZoomData(
            chromosome.name, chromosome.name, 
            'observed', 'KR', 'BP', resolution                                          
        )
    except:
        try: 
            chromosome_matrix = hic.getMatrixZoomData(
                chromosome.name, chromosome.name, 
                'observed', 'SCALE', 'BP', resolution                                          
            )
        except:
            print('Chromosome {} doesnt contain any informative rows'.format(name))
            return 

    informative_indexes = np.array(chromosome_matrix.getNormVector(index))
    informative_indexes = np.where(np.isnan(informative_indexes)^True)[0]

    if len(informative_indexes) == 0:
        print('Chromosome {} doesnt contain any informative rows'.format(name))
        return

    results = chromosome_matrix.getRecords(0, length, 0, length)
    
    # Bottleneck step
    results = np.array([[(r.binX//resolution), (r.binY//resolution), r.counts] for r in results])

    N = length//resolution + 1
    
    mat = csr_matrix((results[:, 2], (results[:, 0], results[:, 1])), shape=(N,N))
    mat = csr_matrix.todense(mat)
    mat = mat.T
    mat = mat + np.tril(mat, -1).T

    np.savez_compressed(output_path, hic=mat, compact=informative_indexes, size=length)
    print('Saving Chromosome at path {}'.format(output_path))
    return True

    
    
def parse_hic_file(path_to_hic_file, output, PARAMETERS, debug=False):
    '''
        This function provides a wrapper on all the methods that 
        reads the .hic file and stores individual chromosomes in a 
        dense matrix format at provided location
        @params: path_to_hic_file <os.path>, path to where hic file is stored
        @params: output_directory <os.path>, path where to store the generated chromsomes
        @params: resolution <int>, resolution at which we sample the HiC contacts, defaults to 10000
        @returns: None
    '''
    print('Parsing out intra-chromosomal contact matrices from {} file.'.format(path_to_hic_file))
    # Read the hic file into memory
    hic = hicstraw.HiCFile(path_to_hic_file)
    
    if PARAMETERS['resolution'] not in hic.getResolutions():
        print('Resolution not supported by the provided .hic file, try a resolution from list {}'.format(
            hic.getResolutions()
        ))
        exit(1)
    
    # Get all the available chromosomes
    chromosomes = hic.getChromosomes()[1:]

    start_time = time.time()

    if MULTIPROCESSING:
        process_pool = []

        for idx in range(len(chromosomes)):
            p = Process(target=process_chromosome, args=(hic, output, PARAMETERS['resolution'], chromosomes[idx], debug ))
            process_pool.append(p)
            p.start()
        
        for process in process_pool:
            process.join()
    else:
        for idx in range(len(chromosomes)):
            process_chromosome(hic, output, PARAMETERS['resolution'], chromosomes[idx], True)


    end_time = time.time()
    
    print('Parsing took {} seconds!'.format(end_time - start_time))


def parse_bulk_datasets(PARAMETERS):
    bulk_hic_files = list(map(lambda x: os.path.join(MOUSE_RAW_BULK_DATA, x), os.listdir(MOUSE_RAW_BULK_DATA)))
    
    for bulk_hic_file in bulk_hic_files:
        hic_file_name = bulk_hic_file.split('/')[-1].split('.')[0]
        # create a directory to store the numpy matrices
        output_folder = os.path.join(MOUSE_PREPROCESSED_DATA_BULK, hic_file_name)
        create_directory(output_folder)
        
        parse_hic_file(bulk_hic_file, output_folder, PARAMETERS)
        
        
        
def create_motif_track(motif_data, chrsize, PARAMETERS):
    motif_data = motif_data.groupby(['start', 'end']).sum()
    motif_data = motif_data.drop(['chr', 'strand'], axis=1)
    motif_data = motif_data.reset_index()
    
    
    starts = motif_data['start'].to_numpy()
    ends = motif_data['end'].to_numpy()
    reads = motif_data['score'].to_numpy()
    
    size = chrsize // PARAMETERS['resolution'] + 1
    track = np.zeros(size)
    
    for i in range(starts.shape[0]):
        track[starts[i]:ends[i] + 1] += reads[i]/((ends[i]+1) - starts[i])

    return track
    
    

 
def parse_motif_file(motif_file, output, PARAMETERS):
    if 'ctcf' == output.split('/')[-1]:
        # Motif specific pre-processing
        motif_data = pd.read_csv(
            motif_file, sep='\t',
            comment='#',
            names=[
                'chr', 'start', 'end',
                'ukwn', 'strand', 'name', 
                'score', 'p-value', 'q-value', 
                'sequence'
            ]
        )
        motif_data = motif_data.drop(columns=['ukwn', 'name', 'p-value', 'q-value', 'sequence'])
        motif_data['start'] = pd.to_numeric(motif_data["start"])
        motif_data['end'] = pd.to_numeric(motif_data["end"])
    
    elif 'cpg' == output.split('/')[-1]:
        motif_data = pd.read_csv(
            motif_file, sep='\t', skiprows=1,
            comment='#',
            names=[
                'chr', 'start', 'end', 
                'length', 'num_CpG', 'score', 
                'obs_exp_freq'
            ]
        )
        motif_data = motif_data.drop(columns=['length', 'num_CpG', 'obs_exp_freq'])
        
        motif_data['start'] = pd.to_numeric(motif_data["start"]).astype('int')
        motif_data['end'] = pd.to_numeric(motif_data["end"]).astype('int')
        motif_data['strand'] = ['+']*len(motif_data['start'].tolist())
        
    else:
        print('Invalid Motif file')
        exit(1)
    
    # Assuming now we have chr, start, end, strand and a scores
    motif_data['start'] = motif_data['start'].copy().floordiv(PARAMETERS['resolution'])
    motif_data['end'] = motif_data['end'].copy().floordiv(PARAMETERS['resolution'])
    
    positive_strand_motifs = motif_data.loc[motif_data['strand'] == '+']
    negative_strand_motifs = motif_data.loc[motif_data['strand'] == '-']
    
    chrom_sizes = read_chromsizes_file(os.path.join(MOUSE_RAW_DATA, 'chrom.sizes')) 
    
    
    for chromosome, size in chrom_sizes.items():
        output_file = os.path.join(output, '{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
        
        chr_positive_strand_motifs = positive_strand_motifs.loc[positive_strand_motifs['chr'] == chromosome]
        chr_negative_strand_motifs = negative_strand_motifs.loc[negative_strand_motifs['chr'] == chromosome]
        
        positive_track = create_motif_track(chr_positive_strand_motifs, size, PARAMETERS)
        negative_track = create_motif_track(chr_negative_strand_motifs, size, PARAMETERS)
        
        if negative_track.sum() != 0:
            combined_tracks = np.stack((positive_track, negative_track))
        else:
            combined_tracks = positive_track
            combined_tracks = combined_tracks.reshape((1, -1))
        
        print('Saving: {}'.format(output_file))
        
        np.save(output_file, combined_tracks)
    



def parse_motifs_datasets(PARAMETERS):
    motif_files = list(map(lambda x: os.path.join(MOUSE_RAW_MOTIFS_DATA, x), os.listdir(MOUSE_RAW_MOTIFS_DATA)))
    
    for motif_file in motif_files:
        if 'archive' in motif_file:
            continue
        
        motif_file_name = motif_file.split('/')[-1].split('.')[0]
        output_folder = os.path.join(MOUSE_PREPROCESSED_MOTIFS_DATA, motif_file_name)
        create_directory(output_folder)
        
        parse_motif_file(motif_file, output_folder, PARAMETERS)





def preprocess_hires_datasets(PARAMETERS):
    '''
        This function parses both scRNA-seq and scHi-C datasets
    '''
    #parse_hires_scrnaseq_datasets(MOUSE_RAW_DATA_SCRNASEQ, PARAMETERS, MOUSE_PREPROCESSED_DATA_SCRNASEQ)
    parse_hires_scrnaseq_datasets(MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ, PARAMETERS, MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ, True)
    
    # parse_hires_schic_datasets(MOUSE_RAW_DATA_SCHIC, PARAMETERS, MOUSE_PREPROCESSED_DATA_SCHIC)
    parse_hires_schic_datasets(MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC, PARAMETERS, MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC)
    
    parse_bulk_datasets(PARAMETERS)
    parse_motifs_datasets(PARAMETERS)
    
    