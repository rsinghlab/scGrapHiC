from src.format_handler import read_cell_by_gene_matrix, convert_cell_by_gene_to_coordinate_matrix, normalize_cell_by_gene_matrix
from src.visualizations import visualize_hic_contact_matrix
from src.matrix_imputations import matImpute
from src.utils import process_GTF3_file

cell_by_gene_matrix = read_cell_by_gene_matrix('data/mm10/HiRES/scRNA-seq/GSE223917_HiRES_brain.rna.umicount.tsv.gz')
normalize_cell_by_gene_matrix(cell_by_gene_matrix)

#coordinate_matrix = convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_matrix, 'data/mm10/gene_coordinates.csv')

# cooler_files_from_pairix_file(
#     'data/mm10/HiRES/scHi-C/GasaE751001/GSM6998595_GasaE751001.pairs.gz',
#     'data/mm10/HiRES/scHi-C/GasaE751001/',
#     'data/mm10/chrom.sizes'
# )

# matrix =  extract_dense_matrix_from_cooler_file('data/mm10/HiRES/scHi-C/GasaE751001/chr1.cool')
# #matrix = matImpute(matrix, 1, 0.5)
# matrix = matrix[:100, :100]

# visualize_hic_contact_matrix(
# 	matrix, 
# 	'matrix.png',
# 	False
# )
