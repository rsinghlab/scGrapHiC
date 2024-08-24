"""
    The purpose of this file is to test whether we observe the similarity in scHi-C profiles at cell cycle phase
    level of hetrogeneity

"""

from sklearn import manifold
from sklearn import decomposition
from sklearn.metrics import jaccard_score
from src.utils import initialize_parameters_from_args, read_npy_file



from src.globals import *
from src.evaluations import *
from src.visualizations import *
from src.normalizations import *
from src.dataset_creator import *
from src.download_datasets import *
from src.preprocess_datasets import *
from src.pseudobulk import parse_metadata
from src.extract_features import ABcompartments, null_extractor, insulationScore


PARAMETERS = initialize_parameters_from_args()
print(PARAMETERS)

cell_cycle_phase_colors = {
    'Rest': '#000000',
    'G1S': '#00FF00',
    'G2M': '#D81B60',
}


def distance_matrix(X, custom_distance):
    X = np.array(X)
    X = np.squeeze(X)
    
    scores = np.zeros((X.shape[0], X.shape[0]))
    
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            score = custom_distance(X[i], X[j])
            scores[i, j] = score
            scores[j, i] = score

    return scores




def parse_schic_files(schic_files, PARAMETERS, output_path, cell_name='mix late mesenchyme'):
    chrom_sizes = read_chromsizes_file(os.path.join(MOUSE_RAW_DATA, 'chrom.sizes'))    
    cell_directory = os.path.join('/users/gmurtaza/data/gmurtaza/scGrapHiC/higashi/', cell_name)
    
    if not os.path.exists(cell_directory):
        os.makedirs(cell_directory)
    
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
                output_tsv_file = os.path.join(cell_directory, '{}_{}_{}.tsv'.format(cell_name, chromosome, PARAMETERS['resolution']))
                
                chrom_data = pairs_data.loc[(pairs_data['chr1'] == chromosome) & (pairs_data['chr2'] == chromosome)]
                
                # Save the coooler file
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

                chrom_data.to_csv(output_tsv_file, sep='\t', index=False)
                
                

def extract_features_for_chromosome(
        cell_names, Y, chr, PARAMETERS, 
        pbulk, feature_extractor,
        normalize=True, 
        smooth=False, 
        visualize=True
    ):
    
    # Extract scHi-C matrices
    X_files = list(map(
        lambda x: os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, x, '{}_{}.npy'.format(chr, PARAMETERS['resolution'])),
        cell_names
    ))
    X = list(map(
        lambda x: read_npy_file(x),
        X_files
    ))
    
    # Get all unique cell cycle phases
    phases = list(set(Y))
    
    
    # Merge matrices
    new_X = []
    new_Y = []
    for phase in phases:
        phase_indices = [index for index, value in enumerate(Y) if value == phase]
        phase_X = [X[i] for i in phase_indices]
        phase_Y = [Y[i] for i in phase_indices]
        for step in range(0, len(phase_X), pbulk):
            # Aggregate pbulk number samples from phase_X list
            if step+pbulk < len(phase_X):
                pbulk_phase_X = np.sum(np.array(phase_X[step:step+pbulk]), axis=0)
                new_X.append(pbulk_phase_X)
                new_Y.append(phase_Y[step])
            else:
                continue
                # pbulk_phase_X = np.sum(np.array(phase_X[step:len(phase_X)]), axis=0)
                # new_X.append(pbulk_phase_X)
                # new_Y.append(phase_Y[step])
                
    print('After pseudo-bulking we have: {} samples left'.format(len(new_X)))
    print('Size of contact map: {}'.format(new_X[0].shape))
    
    # Normalize
    if normalize:
        new_X = list(map(
        lambda x: library_size_normalization(x, 500000),
        new_X
    ))
        
    # Smooth
    if smooth:        
        new_X = list(map(
            lambda x: smooth_adjacency_matrix(x, 0.5),
            new_X
        ))    
    
    
    if visualize:
        for n in range(len(new_X)):
            visualize_hic_contact_matrix(new_X[n], 'visualizations/feature_evaluations/matrices/{}_{}.png'.format(n, new_Y[n]))

    
    
    if feature_extractor == 'NULL':
        new_X = list(map(
            lambda x: null_extractor(x),
            new_X
        ))
        
        new_X = distance_matrix(new_X, SCC)
        print(new_X)
        
        # Converting it to a dissimilarity measure
        new_X = new_X - 1
        
    elif feature_extractor == 'ABcompartments':
        cgpath = '/users/gmurtaza/data/gmurtaza/scGrapHiC/raw/mm10/motifs/archive/cpg_{}.txt'.format(PARAMETERS['resolution'])
        new_X = list(map(
            lambda x: ABcompartments(x, chr, cgpath, PARAMETERS),
            new_X
        ))
        new_X = distance_matrix(new_X, jaccard_score)
        new_X = new_X - 1
        
        
    elif feature_extractor == 'TADs':
        new_X = list(map(
            lambda x: insulationScore(x, 10*PARAMETERS['resolution'], PARAMETERS['resolution']),
            new_X
        ))
        
        if visualize:
            for n in range(len(new_X)):
                visualize_scnrna_seq_tracks(new_X[n], 'visualizations/feature_evaluations/TADs/{}_{}.png'.format(n, new_Y[n]))
        
        new_X = distance_matrix(new_X, mse)
    else: 
        exit(1)
    
    
    return new_X, new_Y
    


def main(path, stage, feature_extractor, pbulk):
    chromosome = [1] #, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    metadata = pd.read_excel(path)
    metadata = metadata[(metadata['Stage'] == stage)]
    
    # metadata = metadata[(metadata['Dedup contacts'] >= 400000)]
    cell_types, cell_names = parse_metadata(metadata)
    print('Unique cell types: ', len(cell_names), '---',  cell_types)
    print('Unique Cell cycle phases', metadata['Cellcycle phase'].unique())
    
    
    metadata = metadata[(metadata['Celltype'] == cell_types[0])]
    print('Filtered cells',  metadata.shape[0], 'of type: ', cell_types[0])
    
    
    X_cellnames =  metadata['Cellname'].to_list()
    Y = metadata['Cellcycle phase'].to_list() 
    g_one_s_score = metadata['G1S.Score'].to_list() 
    g_two_m_score = metadata['G2M.Score'].to_list() 
    
    
    
    sorted_indices = sorted(range(len(Y)), key=lambda i: Y[i])
    Y = [Y[i] for i in sorted_indices]
    X_cellnames = [X_cellnames[i] for i in sorted_indices]
    
    for i, y in enumerate(Y):
        # print(y)
        
        if y in ['Early-S', 'Mid-S', 'Late-S', 'G1']:
            Y[i] = 'Rest'
        elif y in ['G2', 'M']:
            Y[i] = 'Rest'
        elif y in ['G0']:
            Y[i] = 'Rest'
    
    

    # Get full paths
    X_paths = list(map(
        lambda x: os.path.join(MOUSE_RAW_DATA_SCHIC, '{}.pairs.gz'.format(x)),
        X_cellnames
    ))
    
    # Parse and convert to dense matrices 
    # parse_schic_files(X_paths, PARAMETERS, MOUSE_PREPROCESSED_DATA_SCHIC)
    
    X_chrs = []
    Y_labels = []
    for chr in chromosome:
        X_chr, Y_labels = extract_features_for_chromosome(X_cellnames, Y, 'chr{}'.format(chr), PARAMETERS, feature_extractor=feature_extractor, pbulk=pbulk)
        X_chrs.append(X_chr)
        # print(Y_labels)
        
        
    X_chrs = np.array(X_chrs)
    
    X = np.sum(X_chrs, axis=0)
    
    print(X.shape)
    Y = Y_labels
    
    
    # # Plot the distance matrices as a means to test the similarity
    # sns.heatmap(X, annot=False, cmap='coolwarm', linewidths=0.5, linecolor='gray', cbar=True, xticklabels=Y, yticklabels=Y)
    # plt.savefig('visualizations/feature_evaluations/{}_resolution-{}_pbulk-{}_correlation_plot.png'.format(feature_extractor, PARAMETERS['resolution'], pbulk))
    # plt.close()
    
    # X = torch.from_numpy(X)
    # mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
    # Xt = mds.fit_transform(X)
    
    
    # Y_colored = list(map(
    #     lambda y: cell_cycle_phase_colors[y],
    #     Y
    # ))
    
    # plt.scatter(Xt[:,0], Xt[:,1], c=Y_colored)
    # custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in cell_cycle_phase_colors.values()]
    # plt.legend(custom_legend_handles, cell_cycle_phase_colors.keys(), loc='best')
    
    # plt.xlabel('component 0')
    # plt.ylabel('component 1')
    
    # plt.savefig('visualizations/feature_evaluations/{}_resolution-{}_pbulk-{}_MDS_plot.png'.format(feature_extractor, PARAMETERS['resolution'], pbulk))
    # plt.close()
    
    # if feature_extractor == 'NULL':
    #   # Plot arctan stuff  
    #     y0 = np.mean(Xt[:, 0])
    #     x0 = np.mean(Xt[:, 1])
        
    #     angles = np.arctan2(Xt[:, 0] - y0, Xt[:, 1] - x0)
        
    #     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    #     ax.scatter(angles, np.ones_like(angles), c=Y_colored, cmap='hsv', alpha=0.7)
    #     ax.set_yticklabels([])  # Hide radial ticks
    #     plt.savefig('visualizations/feature_evaluations/{}_resolution-{}_pbulk-{}_MDS_arctan_plot.png'.format(feature_extractor, PARAMETERS['resolution'], pbulk))
    #     plt.close()


main(
    HIRES_EMBRYO_METADATA_FILE,
    stage='EX15',
    feature_extractor='NULL',
    pbulk=201
)








"""
  # Extract scRNA-seq features
    if feature_extractor == 'scRNA-seq':
        scrna_seq_data = read_cell_by_gene_matrix(
            os.path.join(MOUSE_RAW_DATA_SCRNASEQ, 'GSE223917_HiRES_embryo.rna.umicount.tsv.gz')
        )
        df_excluded_first_col = scrna_seq_data.iloc[:, 1:]
        variation = df_excluded_first_col.std(axis=1)
        sorted_rows = variation.sort_values(ascending=False)
        N = 1024
        top_rows_indices = sorted_rows.index[:N]
        scrna_seq_data = scrna_seq_data.loc[top_rows_indices]
        
        X = list(map(
            lambda x: scrna_seq_data[x].values,
            cell_names
        ))

    else:
       
        
"""