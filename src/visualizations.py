import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


def visualize_hic_contact_matrix(contact_matrix, output_file, log=False):	
	if(log == True): 
		contact_matrix = np.log10(contact_matrix)
	np.fill_diagonal(contact_matrix, 0)

	plt.matshow(contact_matrix, cmap=REDMAP)
	plt.axis('off')
	plt.savefig(output_file)

	











