import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from matplotlib.colors import LinearSegmentedColormap

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


def visualize_hic_contact_matrix(contact_matrix, output_file):	
    plt.matshow(contact_matrix, cmap=REDMAP)
    plt.axis('off')
    plt.savefig(output_file)
    plt.close()


def visualize_read_distribution_plot(data, output_path):
    """Function displaying the distribution"""
    
    sns.histplot(data=data[data != 0], bins=15, stat='density', alpha= 1, kde=True,
             edgecolor='white', linewidth=0.5,
             line_kws=dict(color='black', alpha=0.5,
                           linewidth=1.5, label='KDE'))
    plt.gca().get_lines()[0].set_color('black') # manually edit line color due to bug in sns v 0.11.0
    plt.legend(frameon=False)
    
    plt.savefig(output_path)
    plt.close()




def visualize_scnrna_seq_tracks(data, output_file):
    print(data.shape)

    if data.shape[-1] > 1:
        fig, axs = plt.subplots(data.shape[-1])
        X = np.array(range(data.shape[0]))
        
        for i in range(data.shape[-1]):    
            axs[i].plot(X, data[:, i])
    else:
        X = np.array(range(data.shape[0]))
        plt.plot(X, data)
    
    
    plt.savefig(output_file)
    plt.close()







def visualize_generated_tracks(generated, target, output_file):
    X = np.array(range(generated.shape[0]))
    if generated.shape[-1] > 1:
        fig, axs = plt.subplots(generated.shape[-1], 2)
        
        for i in range(generated.shape[-1]):    
            axs[i, 0].plot(X, generated[:, i])
        
        for i in range(target.shape[-1]):    
            axs[i, 1].plot(X, target[:, i])
        
        
    else:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(X, generated)
        ax2.plot(X, target)
    
    plt.savefig(output_file)
    plt.close()







def visualize_generated_hic_contact_matrix(generated, target, output_file):	
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.matshow(generated, cmap=REDMAP)
    ax2.matshow(target, cmap=REDMAP)
    
    # ax1.axis('off')
    # ax2.axis('off')
    plt.savefig(output_file)
    plt.close()
