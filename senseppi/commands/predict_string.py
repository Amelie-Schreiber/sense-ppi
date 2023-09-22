import numpy as np
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
import networkx as nx
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.patches import Rectangle
import argparse
import matplotlib.pyplot as plt
import glob
from ..model import SensePPIModel
from ..utils import *
from ..network_utils import *
from ..esm2_model import add_esm_args, compute_embeddings
from .predict import predict


def main(params):
    label_threshold = params.score / 1000.
    pred_threshold = params.pred_threshold / 1000.

    pairs_file = 'protein.pairs_string.tsv'
    fasta_file = 'sequences.fasta'

    print('Fetching interactions from STRING database...')
    get_interactions_from_string(params.genes, species=params.species, add_nodes=params.nodes,
                                 required_score=params.score, network_type=params.network_type)
    process_string_fasta(fasta_file, min_len=params.min_len, max_len=params.max_len)
    generate_pairs_string(fasta_file, output_file=pairs_file, delete_proteins=params.delete_proteins)

    params.fasta_file = fasta_file
    params.pairs_file = pairs_file
    compute_embeddings(params)

    preds = predict(params)

    # open the actions tsv file as dataframe and add the last column with the predictions
    data = pd.read_csv('protein.pairs_string.tsv', delimiter='\t', names=["seq1", "seq2", "string_label"])
    data['binary_label'] = data['string_label'].apply(lambda x: 1 if x > label_threshold else 0)
    data['preds'] = preds

    print(data.sort_values(by=['preds'], ascending=False).to_string())

    string_ids = {}
    string_tsv = pd.read_csv('string_interactions.tsv', delimiter='\t')[
        ['preferredName_A', 'preferredName_B', 'stringId_A', 'stringId_B']]
    for i, row in string_tsv.iterrows():
        string_ids[row['stringId_A']] = row['preferredName_A']
        string_ids[row['stringId_B']] = row['preferredName_B']

    data_to_save = data.copy()
    data_to_save['seq1'] = data_to_save['seq1'].apply(lambda x: string_ids[x])
    data_to_save['seq2'] = data_to_save['seq2'].apply(lambda x: string_ids[x])
    data_to_save = data_to_save.sort_values(by=['preds'], ascending=False)
    data_to_save.to_csv(params.output + '.tsv', sep='\t', index=False)

    if params.graphs:
        # Create two subpolots but make a short gap between them
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})
        # Plot the predictions as matrix but do not sort the labels
        data_heatmap = data.pivot(index='seq1', columns='seq2', values='preds')
        labels_heatmap = data.pivot(index='seq1', columns='seq2', values='string_label')
        # Produce the list of protein names from sequences.fasta file
        protein_names = [line.strip()[1:] for line in open('sequences.fasta', 'r') if line.startswith('>')]
        # Produce the list of gene names from genes Dataframe
        gene_names = [string_ids[prot] for prot in protein_names]
        # Sort the preds for index and columns based on the order of proteins in the fasta file
        data_heatmap = data_heatmap.reindex(index=protein_names, columns=protein_names)
        labels_heatmap = labels_heatmap.reindex(index=protein_names, columns=protein_names)
        # Replace labels with gene names
        data_heatmap.index = gene_names
        data_heatmap.columns = gene_names
        labels_heatmap.index = gene_names
        labels_heatmap.columns = gene_names

        # Remove genes that are in hparams.delete_proteins
        if params.delete_proteins is not None:
            for protein in params.delete_proteins:
                if protein in data_heatmap.index:
                    data_heatmap = data_heatmap.drop(protein, axis=0)
                    data_heatmap = data_heatmap.drop(protein, axis=1)
                    labels_heatmap = labels_heatmap.drop(protein, axis=0)
                    labels_heatmap = labels_heatmap.drop(protein, axis=1)

        # Make sure that the matrices are symmetric for clustering
        labels_heatmap = labels_heatmap.fillna(value=0)
        labels_heatmap = labels_heatmap + labels_heatmap.T
        np.fill_diagonal(labels_heatmap.values, -1)

        data_heatmap = data_heatmap.fillna(value=0)
        data_heatmap = data_heatmap + data_heatmap.T
        np.fill_diagonal(data_heatmap.values, -1)

        linkages = linkage(labels_heatmap, method='complete', metric='euclidean')
        new_labels = np.argsort(fcluster(linkages, 0.05, criterion='distance'))
        col_order = labels_heatmap.columns[new_labels]
        row_order = labels_heatmap.index[new_labels]
        labels_heatmap = labels_heatmap.reindex(index=row_order, columns=col_order)
        data_heatmap = data_heatmap.reindex(index=row_order, columns=col_order)

        # Fill the upper triangle of labels_heatmap with values from data_heatmap
        labels_heatmap.values[np.triu_indices_from(labels_heatmap.values)] = data_heatmap.values[
            np.triu_indices_from(data_heatmap.values)]
        labels_heatmap.fillna(value=-1, inplace=True)

        cmap = matplotlib.cm.get_cmap('coolwarm').copy()
        cmap.set_bad("black")

        sns.heatmap(labels_heatmap, cmap=cmap, vmin=0, vmax=1,
                    ax=ax1, mask=labels_heatmap == -1,
                    cbar=False, square=True)

        cbar = ax1.figure.colorbar(ax1.collections[0], ax=ax1, location='right', pad=0.15)
        cbar.ax.yaxis.set_ticks_position('right')

        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        ax1.set_ylabel('String interactions', weight='bold', fontsize=18)
        ax1.set_title('Predictions', weight='bold', fontsize=18)
        ax1.yaxis.tick_right()
        for i in range(len(labels_heatmap)):
            ax1.add_patch(Rectangle((i, i), 1, 1, fill=True, color='white', alpha=1, zorder=100))

        pred_graph = nx.Graph()

        for i, row in data.iterrows():
            if row['string_label'] > label_threshold:
                pred_graph.add_edge(row['seq1'], row['seq2'], color='black', weight=row['string_label'], style='dotted')
            if row['preds'] > pred_threshold and pred_graph.has_edge(row['seq1'], row['seq2']):
                pred_graph[row['seq1']][row['seq2']]['style'] = 'solid'
                pred_graph[row['seq1']][row['seq2']]['color'] = 'limegreen'
            if row['preds'] > pred_threshold and row['string_label'] <= label_threshold:
                pred_graph.add_edge(row['seq1'], row['seq2'], color='red', weight=row['preds'], style='solid')

        for node in pred_graph.nodes():
            pred_graph.nodes[node]['color'] = 'lightgrey'

        # Replace the string ids with gene names
        pred_graph = nx.relabel_nodes(pred_graph, string_ids)

        pos = nx.spring_layout(pred_graph, k=2., iterations=100)
        nx.draw(pred_graph, pos=pos, with_labels=True, ax=ax2,
                edge_color=[pred_graph[u][v]['color'] for u, v in pred_graph.edges()],
                width=[pred_graph[u][v]['weight'] for u, v in pred_graph.edges()],
                style=[pred_graph[u][v]['style'] for u, v in pred_graph.edges()],
                node_color=[pred_graph.nodes[node]['color'] for node in pred_graph.nodes()])

        legend_elements = [
            Line2D([0], [0], marker='_', color='limegreen', label='PP', markerfacecolor='limegreen', markersize=10),
            Line2D([0], [0], marker='_', color='red', label='FP', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='_', color='black', label='FN - based on STRING', markerfacecolor='black',
                   markersize=10, linestyle='dotted')]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 0.0), ncol=1, fontsize=8)

        save_path = '{}_graph_{}_{}.pdf'.format(params.output, '_'.join(params.genes), params.species)
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print("The graphs were saved to: ", save_path)
        plt.show()
        plt.close()

    os.remove(fasta_file)
    os.remove(pairs_file)
    for f in glob.glob('{}.protein.sequences*'.format(params.species)):
        os.remove(f)
    os.remove('string_interactions.tsv')


def add_args(parser):
    parser = add_general_args(parser)

    string_pred_args = parser.add_argument_group(title="General options")
    parser._action_groups[0].add_argument("genes", type=str, nargs="+",
                                          help="Name of gene to fetch from STRING database. Several names can be "
                                               "typed (separated by whitespaces).")
    string_pred_args.add_argument("--model_path", type=str, default=None,
                                  help="A path to .ckpt file that contains weights to a pretrained model. If "
                                       "None, the preinstalled senseppi.ckpt trained version is used. "
                                       "(Trained on human PPIs)")
    string_pred_args.add_argument("-s", "--species", type=int, default=9606,
                                  help="Species from STRING database. Default: H. Sapiens")
    string_pred_args.add_argument("-n", "--nodes", type=int, default=10,
                                  help="Number of nodes to fetch from STRING database. ")
    string_pred_args.add_argument("-r", "--score", type=int, default=0,
                                  help="Score threshold for STRING connections. Range: (0, 1000). ")
    string_pred_args.add_argument("-p", "--pred_threshold", type=int, default=500,
                                  help="Prediction threshold. Range: (0, 1000). ")
    string_pred_args.add_argument("--graphs", action='store_true',
                                  help="Enables plotting the heatmap and a network graph.")
    string_pred_args.add_argument("-o", "--output", type=str, default="preds_from_string",
                                  help="A path to a file where the predictions will be saved. "
                                       "(.tsv format will be added automatically)")
    string_pred_args.add_argument("--network_type", type=str, default="physical", choices=['physical', 'functional'],
                                  help="Network type to fetch from STRING database. ")
    string_pred_args.add_argument("--delete_proteins", type=str, nargs="+", default=None,
                                  help="List of proteins to delete from the graph. "
                                       "Several names can be specified separated by whitespaces. ")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)
    return parser


if __name__ == '__main__':
    pred_parser = argparse.ArgumentParser()
    pred_parser = add_args(pred_parser)
    pred_params = pred_parser.parse_args()

    main(pred_params)
