import os
import re
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from src.globals import RESULTS, DATASET_LABELS_JSON
from src.utils import create_directory
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


def visualize_hic_contact_matrix(contact_matrix, output_file):
    plt.matshow(contact_matrix, cmap=REDMAP)
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def visualize_read_distribution_plot(data, output_path):
    sns.histplot(data=data[data != 0], bins=15, stat='density', alpha= 1, kde=True,
             edgecolor='white', linewidth=0.5,
             line_kws=dict(color='black', alpha=0.5,
                           linewidth=1.5, label='KDE'))
    plt.gca().get_lines()[0].set_color('black') # manually edit line color due to bug in sns v 0.11.0
    plt.legend(frameon=False)
    
    plt.savefig(output_path)
    plt.close()




def visualize_scnrna_seq_tracks(data, output_file):
    
    if len(data.shape) >= 2 and data.shape[-1] > 1:
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
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def log_results(generated, target, score, idx, mtd, PARAMETERS):
    generated = generated.detach().to('cpu').numpy()
    target = target.detach().to('cpu').numpy()
    idx = idx.detach().to('cpu').numpy()
    mtd = mtd.detach().to('cpu').numpy()
    
    with open(DATASET_LABELS_JSON, 'r') as openfile:
        # Reading from json file
        dataset_object = json.load(openfile)
        stage_dict = {value: key for key, value in dataset_object['stage'].items()}
        tissue_dict = {value: key for key, value in dataset_object['tissue'].items()}
        cell_type_dict = {value: key for key, value in dataset_object['cell_type'].items()}
        
    output_folder = os.path.join(
        RESULTS, PARAMETERS['experiment'], 
        stage_dict[mtd[0]], 
        tissue_dict[mtd[1]],
        cell_type_dict[mtd[2]], 
        str(mtd[3]),
    )
    
    create_directory(output_folder)
    create_directory(os.path.join(output_folder, 'visualizations'))
    create_directory(os.path.join(output_folder, 'generated'))
    create_directory(os.path.join(output_folder, 'targets'))
    
    results_file = os.path.join(RESULTS, PARAMETERS['experiment'], 'results.csv')

    file_name = 'chr{}_s{}_e{}'.format(
        idx[0], idx[2], idx[3]
    )
    
    visualize_generated_hic_contact_matrix(
        generated, target,
        os.path.join(output_folder, 'visualizations', '{}.png'.format(file_name))
    )
    np.save(
        os.path.join(output_folder, 'generated', '{}.npy'.format(file_name)),
        generated
    )
    np.save(
        os.path.join(output_folder, 'targets', '{}.npy'.format(file_name)),
        target
    )
    
    with open(results_file, 'a+') as f:
        f.write(
            '{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                tissue_dict[mtd[1]],
                stage_dict[mtd[0]],
                cell_type_dict[mtd[2]],
                mtd[3],
                'chr{}'.format(idx[0]),
                idx[2],
                idx[3],
                score[0],
                score[1],
                score[2],
                score[3]
            )
        )
    
    
    


def plot_num_cell_to_performance_scatter_plot(results_file, metric, output):
    data = pd.read_csv(
        results_file, sep=',',
        names=[
            'tissue', 'stage', 'cell_type',
            'cell_count', 'chr', 'start_0', 'start_1', 
            'MSE', 'SSIM', 'GD', 'SCC'
        ]
    )
    
    X  = data['cell_count'].unique()
    Y = []
    stds = []
    print(X)
    
    for cell_count in X:
        temp_df = data.loc[data['cell_count'] == cell_count] 
        Y.append(temp_df[[metric]].mean()[metric])
        stds.append(temp_df[[metric]].std()[metric])
    
    
    coefficients = np.polyfit(X, Y, 2)
    poly_fit = np.poly1d(coefficients)
    
    x_values = np.linspace(min(X), max(X), 100)

    # Calculate corresponding y values using the polynomial fit
    y_values = poly_fit(x_values)

    plt.scatter(X, Y)
    plt.plot(x_values, y_values, color='steelblue', linestyle='--')
   

    plt.xlabel('cell count')
    plt.ylabel(metric)
    
    plt.savefig(output, bbox_inches='tight')
    plt.close()
    
    
def plot_grouped_boxplot(results_file, output):
    data = pd.read_csv(
        results_file, sep=',',
        names=[
            'tissue', 'stage', 'cell_type',
            'cell_count', 'chr', 'start_0', 'start_1', 
            'MSE', 'SSIM', 'Kendall_Tau', 'SCC'
        ]
    )
    
    data['name'] = data['tissue'] + '_' + data['stage']
    
    sns.set(rc={'figure.figsize':(8.27,  3)})
    
    sns.boxplot(
        x="name", y="SSIM",
        hue="cell_type", data=data
    )
    plt.gca().get_lines()[0].set_color('black') # manually edit line color due to bug in sns v 0.11.0
    plt.legend(frameon=False)
    plt.ylim(0, 1)

    plt.savefig(output)
    plt.close()


def parse_results(results_file):
    data = pd.read_csv(
        results_file, sep=',',
        names=[
            'tissue', 'stage', 'cell_type',
            'cell_count', 'chr', 'start_0', 'start_1', 
            'SSIM', 'GD', 'SCC', 'TAD_sim', 'MSE', 'Kendall_Tau'
        ]
    )
    # print(data['MSE'].mean())

    return (
        data['MSE'].median(), 
        data['SSIM'].median(),
        data['Kendall_Tau'].median(),
        data['SCC'].median(), 
        data['GD'].median(),
        data['TAD_sim'].median(),
    )


def extract_results(results_file, metric, exclusion, cell_type=''):
    data = pd.read_csv(
        results_file, sep=',',
        names=[
            'tissue', 'stage', 'cell_type',
            'cell_count', 'chr', 'start_0', 'start_1', 
            'SSIM', 'GD', 'SCC', 'TAD_sim', 'MSE', 'Kendall_Tau'
        ]
    )
    data = data[(~data.tissue.isin(exclusion['tissue']))]
    data = data[(~data.stage.isin(exclusion['stage']))]
    data = data[(~data.cell_type.isin(exclusion['cell_line']))]
    
    if cell_type != '':
        data = data[data.cell_type.isin([cell_type])]
    
    return data[metric].to_list()
    
    
    


def create_plots_for_figure2():
    exclusion_set = {
        'tissue': ['brain'], 
        'stage': ['EX15'],
        'cell_line': []
    }
    
    metrics = ['GD', 'SCC', 'TAD_sim']
    
    result_files = [
        # os.path.join(RESULTS, 'bulk_only', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_only', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_ctcf', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg', 'full_results.csv'),
        os.path.join(RESULTS, 'mesc-new', 'full_results.csv'),
    ]
    
    results_titles = [
        # 'Bulk',
        'scRNA-seq',
        'scRNA-seq\n+CTCF',
        'scRNA-seq\n+CTCF+CpG',
        'scGrapHiC'
    ]
    colors = [
        # '#D81B60',
        '#AFA26F',
        '#FFC107',
        '#004D40',
        '#6BBDE7'
    ]
    
    fig, axs = plt.subplots(len(metrics), figsize=(2*len(colors), 12))
    
    print(len(axs))
    
    
    for ax_i, metric in enumerate(metrics):
        results = []
        for result_file in result_files:
            results.append(extract_results(result_file, metric, exclusion_set))
        
        
        positions = np.arange(1, len(results)+1)
        
        
        arrow_params = dict(arrowstyle='|-|', lw=1, color='gray', mutation_scale=2)
        p_values = []
        for i in range(1, len(results)):
            stars = ''
            p_value = ttest_ind(results[i-1], results[i]).pvalue
            if p_value < 0.0001:
                stars+= '*'
            if stars == '':
                stars = '-'
                
            p_values.append(stars)
            
            y_pos_for_the_anotate = 0.85 + i*0.075
            arrow = FancyArrowPatch((positions[i-1]-0.025, y_pos_for_the_anotate), (positions[i]+0.025, y_pos_for_the_anotate), **arrow_params)
            
            # axs[ax_i].text((positions[i-1] + positions[i])/2, y_pos_for_the_anotate+0.005, stars, ha='center', va='bottom', color='black', fontsize=12)
            # axs[ax_i].add_patch(arrow)
            
        
        p_values = ['-'] + p_values
        
        print(ax_i, metric)
        
        vp = axs[ax_i].violinplot(results, showmeans=True, vert=True, widths=0.7, positions=positions, showextrema=None, bw_method=0.5)

        
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            
        
        for i, p_value in enumerate(p_values):
            axs[ax_i].text(positions[i], np.mean(results[i]), f'{np.mean(results[i]):.4f}', ha='center', va='bottom', color='black', fontsize=12)
        
        axs[ax_i].set_ylabel(metric, weight='bold', fontsize=15)
        
        min_val = 10000
        for result in results:
            if min(result) < min_val:
                min_val = min(result)
        
        axs[ax_i].set_ylim(min_val, 1.1)
        axs[ax_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[ax_i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        
        
        
        
        
        axs[ax_i].set_xticks(positions)
        if metric == 'GD':
            axs[ax_i].set_xticklabels(results_titles, rotation=0,  weight='bold', fontsize=15)
        else:
            axs[ax_i].set_xticklabels(results_titles, rotation=0,  weight='bold', fontsize=15)
        
        # Adding a border around the plot
        axs[ax_i].spines['top'].set_bounds(False)
        axs[ax_i].spines['right'].set_color('#CCCCCC')
        axs[ax_i].spines['bottom'].set_color('#CCCCCC')
        axs[ax_i].spines['left'].set_color('#CCCCCC')

    # We need to add lines corresponding to biological maxima TAD_sim = 0.88, GD = 0.94, SCC = 0.79
    background_positions = [0.6, len(results)+0.25]
    
    axs[0].plot(background_positions, [0.94]*len(background_positions), linestyle='dashed', color='black', alpha=0.75)
    axs[1].plot(background_positions, [0.84]*len(background_positions), linestyle='dashed', color='black', alpha=0.75)
    axs[2].plot(background_positions, [0.96]*len(background_positions), linestyle='dashed', color='black', alpha=0.75)   
    
    
    plt.savefig('visualizations/attempt_4_fig2.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()










def create_plots_for_figure3():
    exclusion_set = {
        'tissue': ['brain'], 
        'stage': ['EX15'],
        'cell_line': []
    }
    
    metrics = ['GD', 'SCC', 'TAD_sim']
    
    result_files = [
        os.path.join(RESULTS, 'pn5_zygote', 'full_results.csv'),
        os.path.join(RESULTS, 'early_two_cell', 'full_results.csv'),
        os.path.join(RESULTS, 'eight_cells', 'full_results.csv'),
        os.path.join(RESULTS, 'inner_cell_mass', 'full_results.csv'),
        os.path.join(RESULTS, 'mesc-new', 'full_results.csv'),
    ]
    
    background_result_file =  os.path.join(RESULTS, 'rna_seq_ctcf_cpg', 'full_results.csv')
    
    print(background_result_file)
    
    results_titles = [
        'Zygote\n(E0.5)',
        'Two Cells\n(E1.5)',
        'Eight Cells\n(E2.5)',
        'ICM\n(E3.5)',
        'mESC\n(E4.5)',
    ]
    
    colors = [
        '#D81B60',
        '#AFA26F',
        '#004D40',
        '#6D026A',
        '#6BBDE7'
    ]
    
    fig, axs = plt.subplots(len(metrics), sharex=True, figsize=(9, 12))
    
    print(len(axs))
    
    
    for ax_i, metric in enumerate(metrics):
        results = []
        for result_file in result_files:
            results.append(extract_results(result_file, metric, exclusion_set))
        
        background_result = extract_results(background_result_file, metric, exclusion_set)
        positions = np.arange(1, len(results)+1)
        
        arrow_params = dict(arrowstyle='|-|', lw=1, color='gray', mutation_scale=2)
        p_values = []
        for i in range(1, len(results)):
            stars = ''
            p_value = ttest_ind(results[i-1], results[i]).pvalue
            if p_value < 0.0001:
                stars+= '*'
            if stars == '':
                stars = '-'
                
            p_values.append(stars)
            
            y_pos_for_the_anotate = 0.85 + i*0.075
            arrow = FancyArrowPatch((positions[i-1]-0.025, y_pos_for_the_anotate), (positions[i]+0.025, y_pos_for_the_anotate), **arrow_params)
            
            # axs[ax_i].text((positions[i-1] + positions[i])/2, y_pos_for_the_anotate+0.005, stars, ha='center', va='bottom', color='black', fontsize=12)
            # axs[ax_i].add_patch(arrow)
        
        p_values = ['-'] + p_values
        
        print(ax_i, metric)
        
        vp = axs[ax_i].violinplot(results, showmeans=True, vert=True, widths=0.7, positions=positions, showextrema=None, bw_method=0.5)
        
        background_positions = [0.6, len(results)+0.25]
        
        axs[ax_i].plot(background_positions, [np.mean(background_result)]*len(background_positions), linestyle='dashed', color='gray', alpha=0.75)
        
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            
        
        for i, p_value in enumerate(p_values):
            axs[ax_i].text(positions[i], np.mean(results[i]), f'{np.mean(results[i]):.4f}', ha='center', va='bottom', color='black', fontsize=12)
        
        axs[ax_i].set_ylabel(metric, weight='bold', fontsize=15)
        
        min_val = 10000
        for result in results:
            if min(result) < min_val:
                min_val = min(result)
        
        axs[ax_i].set_ylim(min_val, 1.2)
        axs[ax_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[ax_i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        
        axs[ax_i].set_xticks(positions)
        if metric == 'TAD_sim':
            axs[ax_i].set_xticklabels(results_titles, rotation=0,  weight='bold', fontsize=15)
        else:
            axs[ax_i].set_xticklabels(results_titles, rotation=0,  weight='bold', fontsize=15)
        
        # Adding a border around the plot
        axs[ax_i].spines['top'].set_bounds(False)
        axs[ax_i].spines['right'].set_color('#CCCCCC')
        axs[ax_i].spines['bottom'].set_color('#CCCCCC')
        axs[ax_i].spines['left'].set_color('#CCCCCC')

    # We need to add lines corresponding to biological maxima TAD_sim = 0.88, GD = 0.94, SCC = 0.79
    background_positions = [0.6, len(results)+0.25]
    
    axs[0].plot(background_positions, [0.94]*len(background_positions), linestyle='dashed', color='black', alpha=0.75)
    axs[1].plot(background_positions, [0.84]*len(background_positions), linestyle='dashed', color='black', alpha=0.75)
    axs[2].plot(background_positions, [0.96]*len(background_positions), linestyle='dashed', color='black', alpha=0.75) 
        
    plt.savefig('visualizations/fig3.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()





def create_plots_for_figure4():
    background_exclusion_set = {
        'tissue': ['brain'], 
        'stage': ['EX15'],
        'cell_line': []
    }
    
    
    exclusion_set_ex1 = {
        'tissue': [], 
        'stage': ['E70', 'E75', 'E80', 'E85', 'E95', 'EX05'],
        'cell_line': ['mix_late_mesenchyme', 'early_neurons']
    }
    
    exclusion_set_mlm = {
        'tissue': [], 
        'stage': ['E70', 'E75', 'E80', 'E85', 'E95', 'EX05'],
        'cell_line': ['Ex1', 'early_neurons']
    }
    
    exclusion_set_en = {
        'tissue': [], 
        'stage': ['E70', 'E75', 'E80', 'E85', 'E95', 'EX05'],
        'cell_line': ['Ex1', 'mix_late_mesenchyme']
    }
    
    metrics = ['GD', 'SCC', 'TAD_sim']
    
    result_files = [
        os.path.join(RESULTS, 'mesc-new', 'full_results.csv'),
        os.path.join(RESULTS, 'cerebral-cortex', 'full_results.csv'),
    ]
    
    results_titles = [
        'Mix Late\nMesenchyme',
        'Early\nNeurons',
        'Ex1\n(mESC prior)',
        'Ex1\n(Cortex prior)',
    ]
    colors = [
        '#D81B60',
        '#AFA26F',
        '#FFC107',
        '#004D40',
    ]
    
    background_result_file =  os.path.join(RESULTS, 'mesc-new', 'full_results.csv')
    
    
    fig, axs = plt.subplots(len(metrics), sharex=True, figsize=(9, 10))
    
    print(len(axs))
    
    
    for ax_i, metric in enumerate(metrics):
        data = [
            extract_results(result_files[0], metric, exclusion_set_mlm),
            extract_results(result_files[0], metric, exclusion_set_en),
            extract_results(result_files[0], metric, exclusion_set_ex1),
            extract_results(result_files[1], metric, exclusion_set_ex1)
        ]
        background_result = extract_results(background_result_file, metric, background_exclusion_set)
        
        positions = np.arange(1, len(data)+1)
        vp = axs[ax_i].violinplot(data, showmeans=True, vert=True, widths=0.7, positions=positions, showextrema=None, bw_method=0.5)
        
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            
        
        axs[ax_i].set_ylabel(metric, weight='bold', fontsize=15)
        axs[ax_i].set_ylim(min(min(data))-0.1, 1.1)
        axs[ax_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        
        x_points = [2.5, 2.5]
        y_points = [min(min(data))-0.1, 1]
        
        axs[ax_i].plot(x_points, y_points, linestyle='dotted', color='black', linewidth=4)
        
        background_positions = [0.6, len(data)+0.25]
        axs[ax_i].plot(background_positions, [np.mean(background_result)]*len(background_positions), linestyle='dashed', color='gray', alpha=0.75)
        
        
        for i in range(len(data)):
            axs[ax_i].text(positions[i], np.mean(data[i]), f'{np.mean(data[i]):.4f}', ha='center', va='bottom', color='black', fontsize=12)
        
        
        axs[ax_i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        
        axs[ax_i].set_xticks(positions)
        
        if metric == 'TAD_sim':
            axs[ax_i].set_xticklabels(results_titles, rotation=0,  weight='bold', fontsize=15)
        else:
            axs[ax_i].set_xticklabels([' ']*len(results_titles), rotation=0,  weight='bold', fontsize=15)    
        
        
        p_value = ttest_ind(data[-1], data[-2]).pvalue
        stars = ''
        if p_value < 0.0001:
            stars+= '*'
        if stars == '':
            stars = '-'
        
        if metric == 'GD':
            axs[ax_i].text(1.5, 1.1, 'Embryo, Stage EX15', ha='center', va='bottom', weight='bold', color='black', fontsize=14)
            axs[ax_i].text(3.5, 1.1, 'Brain', ha='center', va='bottom', color='black', weight='bold', fontsize=14)
        
        
        # annotate the significant changes
        arrow_params = dict(arrowstyle='|-|', lw=1, color='gray', mutation_scale=2)
        y_pos_for_the_anotate = 1.05
        
        arrow = FancyArrowPatch((positions[-2], y_pos_for_the_anotate), (positions[-1], y_pos_for_the_anotate), **arrow_params)
        
        # axs[ax_i].text((positions[-2] + positions[-1])/2, y_pos_for_the_anotate+0.005, stars, ha='center', va='bottom', color='black', fontsize=12)
        # axs[ax_i].add_patch(arrow)
        
        # Adding a border around the plot
        axs[ax_i].spines['top'].set_bounds(False)
        axs[ax_i].spines['right'].set_color('#CCCCCC')
        axs[ax_i].spines['bottom'].set_color('#CCCCCC')
        axs[ax_i].spines['left'].set_color('#CCCCCC')
        
    plt.savefig('visualizations/fig4.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()
    
        

        
    
def create_hic_visualization_plot_for_figure2():
    stage = 'E70'
    tissue = 'embryo'
    cell_line = 'epiblast_and_PS'
    num_cells = '194'
    file = 'chr7_s640_e640.npy'

    result_files = [
        os.path.join(RESULTS, 'bulk_only',    stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'rna_seq_only',    stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'rna_seq_ctcf',    stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg',stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'mesc-new',        stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'mesc-new',        stage, tissue, cell_line, num_cells, 'targets'  , file),
    ]
    
    result_labels = [
        'Bulk only',
        'scRNA-seq',
        'scRNA-seq+\nCTCF',
        'scRNA-seq+\nCTCF+CpG',
        'scGrapHiC',
        'Target'
    ]
    
    fig, axs = plt.subplots(1, len(result_labels),  sharey=True, figsize=(7, 4))
    
    for i in range(len(result_labels)):
        data = np.load(result_files[i])
        axs[i].matshow(data, cmap=REDMAP)
        axs[i].axis('off')
        axs[i].text(64, -3, result_labels[i], ha='center', va='bottom', weight='bold', color='black', fontsize=7)
        
        rect_0 = plt.Rectangle((-0.5, -0.5), data.shape[1], data.shape[0], linewidth=1, edgecolor='black', facecolor='none')
        rect_1 = plt.Rectangle((57, 57), 43, 43, linewidth=0.675, edgecolor='blue', facecolor='none')
        axs[i].add_patch(rect_0)
        axs[i].add_patch(rect_1)

            
    plt.savefig('visualizations/fig2_hic_visualizations.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()
    
    
    
    
def create_hic_visualization_plot_for_figure3():
    stage = 'E70'
    tissue = 'embryo'
    cell_line = 'epiblast_and_PS'
    num_cells = '194'
    file = 'chr7_s640_e640.npy'
    
    result_files = [
        os.path.join(RESULTS, 'pn5_zygote',    stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'early_two_cell',    stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'eight_cells',        stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'inner_cell_mass',        stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'mesc-new',        stage, tissue, cell_line, num_cells, 'generated', file),
        os.path.join(RESULTS, 'mesc-new',        stage, tissue, cell_line, num_cells, 'targets'  , file),
    ]
    
    result_labels = [
        'Zygote(E0.5)',
        'Two Cells(E1.5)',
        'Eight Cells(E2.5)',
        'ICM(E3.5)',
        'mESC(E4.5)',
        'Target'
    ]
    
    fig, axs = plt.subplots(1, len(result_labels),  sharey=True, figsize=(7, 4))
    
    for i in range(len(result_labels)):
        data = np.load(result_files[i])
        axs[i].matshow(data, cmap=REDMAP)
        axs[i].axis('off')
        axs[i].text(64, -3, result_labels[i], ha='center', va='bottom', weight='bold', color='black', fontsize=7)
        
        rect_0 = plt.Rectangle((-0.5, -0.5), data.shape[1], data.shape[0], linewidth=1, edgecolor='black', facecolor='none')
        rect_1 = plt.Rectangle((10, 10), 51, 51, linewidth=0.675, edgecolor='blue', facecolor='none')
        axs[i].add_patch(rect_0)
        axs[i].add_patch(rect_1)

            
    plt.savefig('visualizations/fig3_hic_visualizations.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()
    

def create_hic_visualization_plot_for_figure4(file):
    result_files = [
        # MLM files
        os.path.join(RESULTS, 'mesc-new',           'EX15', 'embryo', 'mix_late_mesenchyme', '403', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'EX15', 'embryo', 'mix_late_mesenchyme', '403', 'targets', file),
        # EN Files
        os.path.join(RESULTS, 'mesc-new',           'EX15', 'embryo', 'early_neurons', '255', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'EX15', 'embryo', 'early_neurons', '255', 'targets', file),
        # EX1 -- mESC
        os.path.join(RESULTS, 'mesc-new',           'brain', 'brain', 'Ex1', '204', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'brain', 'brain', 'Ex1', '204', 'targets', file),
        # EX1 -- cortex
        os.path.join(RESULTS, 'cerebral-cortex',    'brain', 'brain', 'Ex1', '204', 'generated', file),
        os.path.join(RESULTS, 'cerebral-cortex',    'brain', 'brain', 'Ex1', '204', 'targets', file),
        
    ]
    
    
    fig, axs = plt.subplots(4, 2,  sharey=True, figsize=(2, 3), gridspec_kw = {'wspace':0, 'hspace':0})
    
    
    
    axs[0, 0].matshow(np.load(result_files[0]), cmap=REDMAP)
    axs[0, 0].axis('off')
    axs[0, 0].text(-13, 95, 'Mix Late\nMesenchyme', ha='center', va='bottom', weight='bold', color='black', fontsize=4, rotation=90)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 0].add_patch(rect)
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[0, 0].add_patch(rect_1)
    axs[0, 0].text(64, -3, 'Generated', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    
    axs[0, 1].matshow(np.load(result_files[1]), cmap=REDMAP)
    axs[0, 1].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 1].add_patch(rect)
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[0, 1].add_patch(rect_1)
    axs[0, 1].text(64, -3, 'Target', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    
    
    
    axs[1, 0].matshow(np.load(result_files[2]), cmap=REDMAP)
    axs[1, 0].axis('off')
    axs[1, 0].text(-13, 95, 'Early\nNeurons', ha='center', va='bottom', weight='bold', color='black', rotation=90,  fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[1, 0].add_patch(rect)
    axs[1, 0].add_patch(rect_1)
    
    axs[1, 1].matshow(np.load(result_files[3]), cmap=REDMAP)
    axs[1, 1].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[1, 1].add_patch(rect)
    axs[1, 1].add_patch(rect_1)
    
    
    axs[2, 0].matshow(np.load(result_files[4]), cmap=REDMAP)
    axs[2, 0].axis('off')
    axs[2, 0].text(-13, 95, 'Ex1\n(mESC prior)', ha='center', va='bottom', weight='bold', color='black', rotation=90, fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[2, 0].add_patch(rect)
    axs[2, 0].add_patch(rect_1)
    
    axs[2, 1].matshow(np.load(result_files[5]), cmap=REDMAP)
    axs[2, 1].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[2, 1].add_patch(rect)
    axs[2, 1].add_patch(rect_1)
    
    axs[3, 0].matshow(np.load(result_files[6]), cmap=REDMAP)
    axs[3, 0].axis('off')
    axs[3, 0].text(-13, 95, 'Ex1\n(cortex prior)', ha='center', va='bottom', weight='bold', rotation=90, color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[3, 0].add_patch(rect)
    axs[3, 0].add_patch(rect_1)
    
    axs[3, 1].matshow(np.load(result_files[7]), cmap=REDMAP)
    axs[3, 1].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = plt.Rectangle((57, 57), 30, 30, linewidth=0.675, edgecolor='blue', facecolor='none')
    axs[3, 1].add_patch(rect)
    axs[3, 1].add_patch(rect_1)
    
    plt.tight_layout()
    plt.savefig('visualizations/{}.pdf'.format(file), format='pdf', bbox_inches='tight', dpi=600)
    plt.close()







def create_plot_supp_figure_1():
    exclusion_set = {
        'tissue': ['brain'], 
        'stage': ['EX15'],
        'cell_line': []
    }
    
    metrics = ['GD', 'SCC', 'TAD_sim']
    
    result_files = [
        os.path.join(RESULTS, 'mesc-new', 'full_results.csv'),
    ]
    
    results_titles = [
        'epiblast_and_PS\nStage E70, cells:194',
        'blood\nStage E80, cells:233',
        'early_mesoderm\nStage E75, cells:204',
        'ExE_endoderm\nStage E75, cells:253',
        'ExE_ectoderm\nStage E75, cells:256',
        'neural_ectoderm\nStage E75, cells:390',
        'mix_late_mesenchyme\nStage EX05, cells:391'
    ]
    
    colors = [
        '#D81B60',
        '#AFA26F',
        '#004D40',
        '#6D026A',
        '#6BBDE7',
        '#9A3E4D',
        '#3F3B6F'
    ]
    
    fig, axs = plt.subplots(len(metrics), sharex=True, figsize=(25, 10))
    
    print(len(axs))
    
    
    for ax_i, metric in enumerate(metrics):
        results = []
        for result_file in result_files:
            for result_title in results_titles:
                cell_type = result_title.split('\n')[0]
                results.append(extract_results(result_file, metric, exclusion_set, cell_type))
            
        
        print(ax_i, metric)
        
        #plt.boxplot(results, labels=results_titles)
        positions = np.arange(1, len(results)+1)
        vp = axs[ax_i].violinplot(results, showmeans=True, vert=True, widths=0.7, positions=positions, showextrema=None, bw_method=0.5)

        
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            
        
        for i, p_value in enumerate(results_titles):
            axs[ax_i].text(positions[i], np.mean(results[i]), f'{np.mean(results[i]):.4f}', ha='center', va='bottom', color='black', fontsize=12)
        
        axs[ax_i].set_ylabel(metric, weight='bold', fontsize=15)
        
        min_val = 10000
        for result in results:
            if min(result) < min_val:
                min_val = min(result)
        
        axs[ax_i].set_ylim(min_val, 1)
        axs[ax_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axs[ax_i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        
        axs[ax_i].set_xticks(positions)
        if metric == 'TAD_sim':
            axs[ax_i].set_xticklabels(results_titles, rotation=0,  weight='bold', fontsize=15)
        else:
            axs[ax_i].set_xticklabels([' ']*len(results_titles), rotation=0,  weight='bold', fontsize=15)
        
        # Adding a border around the plot
        axs[ax_i].spines['top'].set_bounds(False)
        axs[ax_i].spines['right'].set_color('#CCCCCC')
        axs[ax_i].spines['bottom'].set_color('#CCCCCC')
        axs[ax_i].spines['left'].set_color('#CCCCCC')

        
    plt.savefig('visualizations/supp_fig_1.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()






def create_plot_supp_figure_2():
    exclusion_set = {
        'tissue': ['brain'], 
        'stage': ['EX15'],
        'cell_line': []
    }
    
    metrics = ['GD', 'SCC', 'TAD_sim']
    
    result_files_line_0 = [
        os.path.join(RESULTS, 'bulk_pdim_two', 'full_results.csv'),
        os.path.join(RESULTS, 'bulk_pdim_four', 'full_results.csv'),
        os.path.join(RESULTS, 'bulk_pdim_eight', 'full_results.csv'),
        os.path.join(RESULTS, 'bulk_pdim_sixteen', 'full_results.csv'),
        os.path.join(RESULTS, 'bulk_pdim_thirty-two', 'full_results.csv'),
    ]
    result_files_line_1 = [
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg_bulk_pdim_two', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg_bulk_pdim_four', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg_bulk_pdim_eight', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg_bulk_pdim_sixteen', 'full_results.csv'),
        os.path.join(RESULTS, 'rna_seq_ctcf_cpg_bulk_pdim_thirty-two', 'full_results.csv'),
    ]
    
    
    results_titles = [
        'bulk',
        'bulk+CTCF+CpG'
    ]
    
    colors = [
        '#D81B60',
        '#AFA26F',
    ]
    
    x_tic_labels = [
        '2', '4', '8', '16', '32'
    ]
    
    fig, axs = plt.subplots(len(metrics), sharex=True, figsize=(13, 9))
    
    print(len(axs))
    
    
    for ax_i, metric in enumerate(metrics):
        l0_results = []
        for result_file in result_files_line_0:
            l0_results.append(np.mean(extract_results(result_file, metric, exclusion_set)))
        
        l1_results = []
        for result_file in result_files_line_1:
            l1_results.append(np.mean(extract_results(result_file, metric, exclusion_set)))
        
        
        
        positions = np.arange(1, len(l0_results)+1)
        
        
        axs[ax_i].plot(positions, l0_results, linewidth=3, markersize=12, color=colors[0],  label=results_titles[0])
        axs[ax_i].plot(positions, l1_results, linewidth=3, markersize=12, color=colors[1],  label=results_titles[1])
        
        
        
        axs[ax_i].set_ylabel(metric, weight='bold', fontsize=15)
        
        axs[ax_i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        
        axs[ax_i].set_xticks(positions)
        if metric == 'TAD_sim':
            axs[ax_i].set_xticklabels(x_tic_labels, rotation=0,  weight='bold', fontsize=15)
            axs[ax_i].set_xlabel('Positional Encoding Dim (k) size', rotation=0,  weight='bold', fontsize=15)
        else:
            axs[ax_i].set_xticklabels([' ']*len(x_tic_labels), rotation=0,  weight='bold', fontsize=15)
        
        if metric == 'GD':
            axs[ax_i].legend()
        
        # Adding a border around the plot
        axs[ax_i].spines['top'].set_bounds(False)
        axs[ax_i].spines['right'].set_color('#CCCCCC')
        axs[ax_i].spines['bottom'].set_color('#CCCCCC')
        axs[ax_i].spines['left'].set_color('#CCCCCC')

        
    plt.savefig('visualizations/supp_fig_2.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.close()




def supporting_visualizations(file):
    print("generating for region:{}".format(file))
    result_files = [
        # MLM files
        os.path.join(RESULTS, 'mesc-new',           'E70',  'embryo', 'epiblast_and_PS', '194', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'E70',  'embryo', 'epiblast_and_PS', '194',  'targets', file),
        
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'early_mesoderm', '204', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'early_mesoderm', '204',  'targets', file),
        
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'ExE_ectoderm', '256', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'ExE_ectoderm', '256',  'targets', file),
        
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'ExE_endoderm', '253', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'ExE_endoderm', '253',  'targets', file),
        
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'neural_ectoderm', '390', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'E75',  'embryo', 'neural_ectoderm', '390',  'targets', file),
        
        os.path.join(RESULTS, 'mesc-new',           'E85',  'embryo', 'blood', '223', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'E85',  'embryo', 'blood', '223',  'targets', file),
        
        os.path.join(RESULTS, 'mesc-new',           'EX05',  'embryo', 'mix_late_mesenchyme', '391', 'generated', file),
        os.path.join(RESULTS, 'mesc-new',           'EX05',  'embryo', 'mix_late_mesenchyme', '391',  'targets', file),
        
    ]
    
    
    fig, axs = plt.subplots(2, 7,  gridspec_kw={'wspace': 0, 'hspace': 0.3})
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    
    axs[0, 0].matshow(np.load(result_files[0]), cmap=REDMAP)
    axs[0, 0].axis('off')
    axs[0, 0].text(64, -3, 'Epiblast and PS', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    axs[0, 0].add_patch(rect)
    axs[0, 0].text(-5, 80, 'Generated', ha='center', va='bottom', weight='bold', color='black', rotation=90, fontsize=4)
    
    
    axs[1, 0].matshow(np.load(result_files[1]), cmap=REDMAP)
    axs[1, 0].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 0].add_patch(rect)
    axs[1, 0].text(-5, 80, 'Target', ha='center', va='bottom', weight='bold', color='black', rotation=90, fontsize=4)
    
    
    axs[0, 1].matshow(np.load(result_files[2]), cmap=REDMAP)
    axs[0, 1].axis('off')
    axs[0, 1].text(64, -3, 'Early Mesoderm', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 1].add_patch(rect)
    
    axs[1, 1].matshow(np.load(result_files[3]), cmap=REDMAP)
    axs[1, 1].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 1].add_patch(rect)
    
    
    axs[0, 2].matshow(np.load(result_files[4]), cmap=REDMAP)
    axs[0, 2].axis('off')
    axs[0, 2].text(64, -3, 'ExE Ectoderm', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 2].add_patch(rect)
    
    axs[1, 2].matshow(np.load(result_files[5]), cmap=REDMAP)
    axs[1, 2].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 2].add_patch(rect)
    
    axs[0, 3].matshow(np.load(result_files[6]), cmap=REDMAP)
    axs[0, 3].axis('off')
    axs[0, 3].text(64, -3, 'ExE Endoderm', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 3].add_patch(rect)
    
    axs[1, 3].matshow(np.load(result_files[7]), cmap=REDMAP)
    axs[1, 3].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 3].add_patch(rect)
    
    

    axs[0, 4].matshow(np.load(result_files[8]), cmap=REDMAP)
    axs[0, 4].axis('off')
    axs[0, 4].text(64, -3, 'Neural Endoderm', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 4].add_patch(rect)
    
    axs[1, 4].matshow(np.load(result_files[9]), cmap=REDMAP)
    axs[1, 4].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 4].add_patch(rect)


    axs[0, 5].matshow(np.load(result_files[10]), cmap=REDMAP)
    axs[0, 5].axis('off')
    axs[0, 5].text(64, -3, 'Blood', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 5].add_patch(rect)
    
    axs[1, 5].matshow(np.load(result_files[11]), cmap=REDMAP)
    axs[1, 5].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 5].add_patch(rect)

    axs[0, 6].matshow(np.load(result_files[12]), cmap=REDMAP)
    axs[0, 6].axis('off')
    axs[0, 6].text(64, -3, 'Mix Late Mesenchyme', ha='center', va='bottom', weight='bold', color='black', fontsize=4)
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[0, 6].add_patch(rect)
    
    axs[1, 6].matshow(np.load(result_files[13]), cmap=REDMAP)
    axs[1, 6].axis('off')
    rect = plt.Rectangle((-0.5, -0.5), np.load(result_files[0]).shape[1], np.load(result_files[0]).shape[0], linewidth=1, edgecolor='black', facecolor='none')
    axs[1, 6].add_patch(rect)
    
    plt.savefig('visualizations/fig4_vis/{}.png'.format(file), bbox_inches='tight', dpi=600)
    plt.close()




def plot_loss_curves():
    log_files = os.listdir('outputs')
    log_files = [x for x in log_files if 'random_seed' in x]
    log_files = list(map(lambda x: os.path.join('outputs', x), log_files))


    for log_file in log_files:
        seed = log_file.split('/')[-1].split('.')[0].split('_')[-1]
        data = open(log_file).read().split('\n')
        data = [x for x in data if 'training/loss' in x]
        data = [x.split('training/loss')[-1] for x in data]
        data = [float(re.search(r'tensor\((\d+\.\d+)\,', x).group(1)) for x in data]

        X = list(range(len(data)))
        plt.plot(X, data, label=seed)
    
    plt.legend()
    plt.ylabel('MSE Loss', weight='bold', )
    plt.xlabel('Batch ID (Across all Epochs)', weight='bold', )

    plt.savefig('visualizations/loss_curve.pdf', format='pdf', dpi=600)
    plt.close()
