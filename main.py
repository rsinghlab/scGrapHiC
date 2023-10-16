from src.format_handler import read_pairix_file, cooler_files_from_pairix_file
from src.utils import read_chromsizes_file, chrom_bins

# chrom_sizes = read_chromsizes_file('data/mm10/chrom.sizes')

# print(chrom_bins('chr1', chrom_sizes['chr1']))

cooler_files_from_pairix_file(
    'data/mm10/HiRES/scHi-C/GasaE751001/GSM6998595_GasaE751001.pairs.gz',
    'data/mm10/HiRES/scHi-C/GasaE751001/',
    'data/mm10/chrom.sizes'
)