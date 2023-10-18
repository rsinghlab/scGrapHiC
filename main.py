from src.format_handler import extract_dense_matrix_from_cooler_file, cooler_files_from_pairix_file
from src.visualizations import visualize_hic_contact_matrix
from src.matrix_imputations import matImpute
from src.utils import process_GTF3_file

process_GTF3_file('data/mm10/gencode.vM23.annotation.gff3.gz')

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
