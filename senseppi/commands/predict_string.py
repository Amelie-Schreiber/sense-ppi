import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
import networkx as nx
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import glob

from ..model import SensePPIModel
from ..utils import *
from ..network_utils import *
from ..esm2_model import add_esm_args, compute_embeddings
from ..dataset import PairSequenceData


def main(params):
    LABEL_THRESHOLD = params.score / 1000.
    PRED_THRESHOLD = params.pred_threshold / 1000.
    pairs_file = 'protein.pairs_string.tsv'
    fasta_file = 'sequences.fasta'

    print('Fetching interactions from STRING database...')
    get_interactions_from_string(params.genes, species=params.species, add_nodes=params.nodes,
                                 required_score=params.score, network_type=params.network_type)
    process_string_fasta(fasta_file, min_len=params.min_len, max_len=params.max_len)
    generate_pairs_string(fasta_file, output_file=pairs_file, with_self=False, delete_proteins=params.delete_proteins)

    params.fasta_file = fasta_file
    compute_embeddings(params)

    test_data = PairSequenceData(emb_dir=params.output_dir_esm, actions_file=pairs_file,
                                 max_len=params.max_len, labels=False)

    pretrained_model = SensePPIModel(params)

    if params.device == 'gpu':
        checkpoint = torch.load(params.model_path)
    elif params.device == 'mps':
        checkpoint = torch.load(params.model_path, map_location=torch.device('mps'))
    else:
        checkpoint = torch.load(params.model_path, map_location=torch.device('cpu'))

    pretrained_model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(accelerator=params.device, logger=False)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=params.batch_size,
                             num_workers=4)

    preds = [pred for batch in trainer.predict(pretrained_model, test_loader) for pred in batch.squeeze().tolist()]
    preds = np.asarray(preds)

    # open the actions tsv file as dataframe and add the last column with the predictions
    data = pd.read_csv('protein.pairs_string.tsv', delimiter='\t', names=["seq1", "seq2", "string_label"])
    data['binary_label'] = data['string_label'].apply(lambda x: 1 if x > LABEL_THRESHOLD else 0)
    data['preds'] = preds

    print(data.sort_values(by=['preds'], ascending=False).to_string())
    data.to_csv(params.output + '.tsv', sep='\t', index=False)

    # Calculate torch metrics based on data['binary_label'] and data['preds']
    torch_labels = torch.tensor(data['binary_label'])
    torch_preds = torch.tensor(data['preds'])
    print('Accuracy: ', Accuracy(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('Precision: ', Precision(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('Recall: ', Recall(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('F1Score: ', F1Score(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('MatthewsCorrCoef: ',
          MatthewsCorrCoef(num_classes=2, threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('ROCAUC: ', AUROC(task='binary')(torch_preds, torch_labels))

    string_ids = {}
    string_tsv = pd.read_csv('string_interactions.tsv', delimiter='\t')[
        ['preferredName_A', 'preferredName_B', 'stringId_A', 'stringId_B']]
    for i, row in string_tsv.iterrows():
        string_ids[row['stringId_A']] = row['preferredName_A']
        string_ids[row['stringId_B']] = row['preferredName_B']

    # This part was needed to color the pairs belonging to the train data, temporarily removed

    # print('Fetching gene names for training set from STRING...')
    #
    # if not os.path.exists('all_genes_train.tsv'):
    #     all_genes = generate_dscript_gene_names(
    #         file_path=actions_path,
    #         only_positives=True,
    #         species=str(hparams.species))
    #     all_genes.to_csv('all_genes_train.tsv', sep='\t', index=False)
    # else:
    #     all_genes = pd.read_csv('all_genes_train.tsv', sep='\t')

    # full_train_data = pd.read_csv(actions_path,
    #                               delimiter='\t', names=['seq1', 'seq2', 'label'])
    #
    # if all_genes is not None:
    #     full_train_data = full_train_data.merge(all_genes, left_on='seq1', right_on='QueryString', how='left').merge(
    #         all_genes, left_on='seq2', right_on='QueryString', how='left')
    #
    #     full_train_data = full_train_data[['preferredName_x', 'preferredName_y', 'label']]
    #
    #     positive_train_data = full_train_data[full_train_data['label'] == 1][['preferredName_x', 'preferredName_y']]
    #     full_train_data = full_train_data[['preferredName_x', 'preferredName_y']]
    #
    #     full_train_data = [tuple(x) for x in full_train_data.values]
    #     positive_train_data = [tuple(x) for x in positive_train_data.values]
    # else:
    #     full_train_data = None
    #     positive_train_data = None

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

        # This part was needed to color the pairs belonging to the train data, temporarily removed

        # if full_train_data is not None:
        #     for i, row in labels_heatmap.iterrows():
        #         for j, _ in row.items():
        #             if (i, j) in full_train_data or (j, i) in full_train_data:
        #                 labels_heatmap.loc[i, j] = -1

        cmap = matplotlib.cm.get_cmap('coolwarm').copy()
        cmap.set_bad("black")

        sns.heatmap(labels_heatmap, cmap=cmap, vmin=0, vmax=1,
                    ax=ax1, mask=labels_heatmap == -1,
                    cbar=False, square=True)  # , linewidths=0.5, linecolor='white')

        cbar = ax1.figure.colorbar(ax1.collections[0], ax=ax1, location='right', pad=0.15)
        cbar.ax.yaxis.set_ticks_position('right')

        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        ax1.set_ylabel('String interactions', weight='bold', fontsize=18)
        ax1.set_title('Predictions', weight='bold', fontsize=18)
        ax1.yaxis.tick_right()
        for i in range(len(labels_heatmap)):
            ax1.add_patch(Rectangle((i, i), 1, 1, fill=True, color='white', alpha=1, zorder=100))

        G = nx.Graph()
        for i, row in data.iterrows():
            if row['string_label'] > LABEL_THRESHOLD:
                G.add_edge(row['seq1'], row['seq2'], color='black', weight=row['string_label'], style='dotted')
            if row['preds'] > PRED_THRESHOLD and G.has_edge(row['seq1'], row['seq2']):
                G[row['seq1']][row['seq2']]['style'] = 'solid'
                G[row['seq1']][row['seq2']]['color'] = 'limegreen'
            if row['preds'] > PRED_THRESHOLD and row['string_label'] <= LABEL_THRESHOLD:
                G.add_edge(row['seq1'], row['seq2'], color='red', weight=row['preds'], style='solid')

        # Replace the string ids with gene names
        G = nx.relabel_nodes(G, string_ids)

        # This part was needed to color the pairs belonging to the train data, temporarily removed

        # if positive_train_data is not None:
        #     for edge in G.edges():
        #         if (edge[0], edge[1]) in positive_train_data or (edge[1], edge[0]) in positive_train_data:
        #             print('TRAINING EDGE: ', edge)
        #             G[edge[0]][edge[1]]['color'] = 'darkblue'
        #             # G[edge[0]][edge[1]]['weight'] = 1

        # Make nodes red if they are present in training data
        for node in G.nodes():
            # if all_genes is not None and node in all_genes['preferredName'].values:
            #     G.nodes[node]['color'] = 'orange'
            # else:
            G.nodes[node]['color'] = 'lightgrey'

        pos = nx.spring_layout(G, k=2., iterations=100)
        nx.draw(G, pos=pos, with_labels=True, ax=ax2,
                edge_color=[G[u][v]['color'] for u, v in G.edges()], width=[G[u][v]['weight'] for u, v in G.edges()],
                style=[G[u][v]['style'] for u, v in G.edges()],
                node_color=[G.nodes[node]['color'] for node in G.nodes()])

        legend_elements = [
            Line2D([0], [0], marker='_', color='darkblue', label='PP from training data', markerfacecolor='darkblue',
                   markersize=10),
            Line2D([0], [0], marker='_', color='limegreen', label='PP', markerfacecolor='limegreen', markersize=10),
            Line2D([0], [0], marker='_', color='red', label='FP', markerfacecolor='red', markersize=10)]
            # Line2D([0], [0], marker='_', color='black', label='FN - based on STRING', markerfacecolor='black',
            #        markersize=10, linestyle='dotted')]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 0.0), ncol=1, fontsize=8)

        savepath = '{}_graph_{}_{}.pdf'.format(params.output, '_'.join(params.genes), params.species)
        plt.savefig(savepath, bbox_inches='tight', dpi=600)
        print("The graphs were saved to: ", savepath)
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
    parser._action_groups[0].add_argument("model_path", type=str,
                                          help="A path to .ckpt file that contains weights to a pretrained model.")
    parser._action_groups[0].add_argument("genes", type=str, nargs="+",
                         help="Name of gene to fetch from STRING database. Several names can be typed (separated by "
                              "whitespaces)")
    string_pred_args.add_argument("-s", "--species", type=int, default=9606,
                         help="Species from STRING database. Default: 9606 (H. Sapiens)")
    string_pred_args.add_argument("-n", "--nodes", type=int, default=10,
                         help="Number of nodes to fetch from STRING database. Default: 10")
    string_pred_args.add_argument("-r", "--score", type=int, default=0,
                         help="Score threshold for STRING connections. Range: (0, 1000). Default: 500")
    string_pred_args.add_argument("-p", "--pred_threshold", type=int, default=500,
                         help="Prediction threshold. Range: (0, 1000). Default: 500")
    string_pred_args.add_argument("--graphs", action='store_true', help="Enables plotting the heatmap and a network graph.")
    string_pred_args.add_argument("-o", "--output", type=str, default="preds_from_string",
                              help="A path to a file where the predictions will be saved. "
                                   "(.tsv format will be added automatically)")
    string_pred_args.add_argument("--network_type", type=str, default="physical",
                         help="Network type: \"physical\" or \"functional\". Default: \"physical\"")
    string_pred_args.add_argument("--delete_proteins", type=str, nargs="+", default=None,
                         help="List of proteins to delete from the graph. Default: None")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    params = parser.parse_args()