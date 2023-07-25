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
from pathlib import Path

from ..model import SensePPIModel
from ..utils import *
from ..network_utils import *
from ..esm2_model import add_esm_args


def main(hparams):
    LABEL_THRESHOLD = hparams.score / 1000.
    PRED_THRESHOLD = params.pred_threshold / 1000.

    test_data = DscriptData(emb_dir='esm_emb_3B', max_len=800, dir_path='', actions_file='protein.actions.tsv')

    actions_path = os.path.join('..', 'Data', 'Dscript', 'preprocessed', 'human_train.tsv')
    loadpath = os.path.join('..', DSCRIPT_PATH)


    model = SensePPIModel(hparams)

    if hparams.nogpu:
        checkpoint = torch.load(loadpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(loadpath)

    model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(accelerator="cpu" if hparams.nogpu else 'gpu', logger=False)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=64,
                             num_workers=4)

    preds = [pred for batch in trainer.predict(model, test_loader) for pred in batch.squeeze().tolist()]
    preds = np.asarray(preds)

    # open the actions tsv file as dataframe and add the last column with the predictions
    data = pd.read_csv('protein.actions.tsv', delimiter='\t', names=["seq1", "seq2", "label"])
    data['binary_label'] = data['label'].apply(lambda x: 1 if x > LABEL_THRESHOLD else 0)
    data['preds'] = preds

    if hparams.normalize:
        data['preds'] = (data['preds'] - data['preds'].min()) / (data['preds'].max() - data['preds'].min())

    print(data.sort_values(by=['preds'], ascending=False).to_string())

    # Calculate torch metrics based on data['binary_label'] and data['preds']
    torch_labels = torch.tensor(data['binary_label'])
    torch_preds = torch.tensor(data['preds'])
    print('Accuracy: ', Accuracy(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('Precision: ', Precision(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('Recall: ', Recall(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('F1Score: ', F1Score(threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('MatthewsCorrCoef: ',
          MatthewsCorrCoef(num_classes=2, threshold=PRED_THRESHOLD, task='binary')(torch_preds, torch_labels))
    print('ROCAUC: ', AUROC()(torch_preds, torch_labels))

    # Create a dictionary of string ids and gene names from string_interactions_short.tsv
    string_ids = {}
    string_tsv = pd.read_csv('string_interactions.tsv', delimiter='\t')[
        ['preferredName_A', 'preferredName_B', 'stringId_A', 'stringId_B']]
    for i, row in string_tsv.iterrows():
        string_ids[row['stringId_A']] = row['preferredName_A']
        string_ids[row['stringId_B']] = row['preferredName_B']

    print('Fetching gene names for training set from STRING...')

    if not os.path.exists('all_genes_train.tsv'):
        all_genes = generate_dscript_gene_names(
            file_path=actions_path,
            only_positives=True,
            species=str(hparams.species))
        all_genes.to_csv('all_genes_train.tsv', sep='\t', index=False)
    else:
        all_genes = pd.read_csv('all_genes_train.tsv', sep='\t')

    # Create a tuple of gene pairs presented in training data, corrresponding gene names are found in 'genes' DataFrame
    full_train_data = pd.read_csv(actions_path,
                                  delimiter='\t', names=['seq1', 'seq2', 'label'])

    # To make sure that we do not use the test species in the training data
    full_train_data = full_train_data[full_train_data.seq1.str.startswith('6239') == False]
    full_train_data = full_train_data[full_train_data.seq2.str.startswith('6239') == False]

    if all_genes is not None:
        full_train_data = full_train_data.merge(all_genes, left_on='seq1', right_on='QueryString', how='left').merge(
            all_genes, left_on='seq2', right_on='QueryString', how='left')

        full_train_data = full_train_data[['preferredName_x', 'preferredName_y', 'label']]

        positive_train_data = full_train_data[full_train_data['label'] == 1][['preferredName_x', 'preferredName_y']]
        full_train_data = full_train_data[['preferredName_x', 'preferredName_y']]

        full_train_data = [tuple(x) for x in full_train_data.values]
        positive_train_data = [tuple(x) for x in positive_train_data.values]
    else:
        full_train_data = None
        positive_train_data = None

    if not hparams.no_graphs:
        # Create two subpolots but make a short gap between them
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})
        # Plot the predictions as matrix but do not sort the labels
        data_heatmap = data.pivot(index='seq1', columns='seq2', values='preds')
        labels_heatmap = data.pivot(index='seq1', columns='seq2', values='label')
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
        if hparams.delete_proteins is not None:
            for protein in hparams.delete_proteins:
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

        # In (labels+data)_heatmap, if a pair of genes is in train data, color it in black in the upper triangle
        if full_train_data is not None:
            for i, row in labels_heatmap.iterrows():
                for j, _ in row.items():
                    if (i, j) in full_train_data or (j, i) in full_train_data:
                        labels_heatmap.loc[i, j] = -1

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
        # ax1.plot([0, len(labels_heatmap)], [0, len(labels_heatmap)], color='white')
        for i in range(len(labels_heatmap)):
            ax1.add_patch(Rectangle((i, i), 1, 1, fill=True, color='white', alpha=1, zorder=100))

        # print(data[data['label'] > 0].sort_values(by=['preds'], ascending=False))

        # Build a multigraph from the data with the predictions as edge weights and edges in red and the labels as secondary edges in black
        G = nx.Graph()
        for i, row in data.iterrows():
            if row['label'] > LABEL_THRESHOLD:
                G.add_edge(row['seq1'], row['seq2'], color='black', weight=row['label'], style='dotted')
            if row['preds'] > PRED_THRESHOLD and G.has_edge(row['seq1'], row['seq2']):
                G[row['seq1']][row['seq2']]['style'] = 'solid'
                G[row['seq1']][row['seq2']]['color'] = 'limegreen'
            if row['preds'] > PRED_THRESHOLD and row['label'] <= LABEL_THRESHOLD:
                G.add_edge(row['seq1'], row['seq2'], color='red', weight=row['preds'], style='solid')

        # Replace the string ids with gene names
        G = nx.relabel_nodes(G, string_ids)

        # If edge is present in training data make it blue
        if positive_train_data is not None:
            for edge in G.edges():
                if (edge[0], edge[1]) in positive_train_data or (edge[1], edge[0]) in positive_train_data:
                    print('TRAINING EDGE: ', edge)
                    G[edge[0]][edge[1]]['color'] = 'darkblue'
                    # G[edge[0]][edge[1]]['weight'] = 1

        # Make nodes red if they are present in training data
        for node in G.nodes():
            if all_genes is not None and node in all_genes['preferredName'].values:
                G.nodes[node]['color'] = 'orange'
            else:
                G.nodes[node]['color'] = 'lightgrey'

        # nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=nx.get_edge_attributes(G, 'weight'))
        # Make edges longer and do not put nodes too close to each other
        pos = nx.spring_layout(G, k=2., iterations=100)

        # Call the same function nx.draw with the same arguments as above but make sure it is plotted in ax2
        nx.draw(G, pos=pos, with_labels=True, ax=ax2,
                edge_color=[G[u][v]['color'] for u, v in G.edges()], width=[G[u][v]['weight'] for u, v in G.edges()],
                style=[G[u][v]['style'] for u, v in G.edges()],
                node_color=[G.nodes[node]['color'] for node in G.nodes()])

        # Put a legend for colors
        legend_elements = [
            Line2D([0], [0], marker='_', color='darkblue', label='PP from training data', markerfacecolor='darkblue',
                   markersize=10),
            Line2D([0], [0], marker='_', color='limegreen', label='PP', markerfacecolor='limegreen', markersize=10),
            Line2D([0], [0], marker='_', color='red', label='FP', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='_', color='black', label='FN - based on STRING', markerfacecolor='black',
                   markersize=10, linestyle='dotted')]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 0.0), ncol=1, fontsize=8)

        savepath = 'graph_{}_{}'.format('_'.join(hparams.genes), hparams.species)
        if 'Dscript' in actions_path:
            savepath += '_Dscript'
            if 'HVLSTM' in loadpath:
                savepath += '_HVLSTM'
        else:
            version = re.search('version_([0-9]+)', loadpath)
            if version is not None:
                savepath += '_v' + version.group(1)

        savepath += '.pdf'
        plt.savefig(savepath, bbox_inches='tight', dpi=600)
        print("The graphs were saved in: ", savepath)
        plt.show()
        plt.close()

def add_args(parser):
    parser = add_general_args(parser)

    parser2 = parser.add_argument_group(title="General options")
    parser2.add_argument("--no_graphs", action='store_true', help="No plotting testing graphs.")
    parser2.add_argument("-g", "--genes", type=str, nargs="+", default="RFC5",
                         help="Name of gene to fetch from STRING database. Several names can be typed (separated by whitespaces). Default: RFC5")
    parser2.add_argument("-s", "--species", type=int, default=9606,
                         help="Species from STRING database. Default: 9606 (H. Sapiens)")
    parser2.add_argument("-n", "--nodes", type=int, default=10,
                         help="Number of nodes to fetch from STRING database. Default: 10")
    parser2.add_argument("-r", "--score", type=int, default=500,
                         help="Score threshold for STRING connections. Range: (0, 1000). Default: 500")
    parser2.add_argument("-p", "--pred_threshold", type=int, default=500,
                         help="Prediction threshold. Range: (0, 1000). Default: 500")
    parser2.add_argument("--network_type", type=str, default="physical",
                         help="Network type: \"physical\" or \"functional\". Default: \"physical\"")
    parser2.add_argument("--normalize", action='store_true', help="Normalize the predictions.")
    parser2.add_argument("--delete_proteins", type=str, nargs="+", default=None,
                         help="List of proteins to delete from the graph. Default: None")

    parser = SensePPIModel.add_model_specific_args(parser)
    remove_argument(parser, "--lr")

    add_esm_args(parser)
    return parser

if __name__ == '__main__':


    params = parser.parse_args()

    if torch.cuda.is_available():
        # torch.cuda.set_per_process_memory_fraction(0.9, 0)
        print('Number of devices: ', torch.cuda.device_count())
        print('GPU used: ', torch.cuda.get_device_name(0))
        torch.set_float32_matmul_precision('high')
    else:
        print('No GPU available, using the CPU instead.')
        params.nogpu = True

    print('Fetching interactions from STRING database...')
    get_interactions_from_string(params.genes, species=params.species, add_nodes=params.nodes,
                                 required_score=params.score, network_type=params.network_type)
    process_string_fasta('sequences.fasta')
    generate_pairs('sequences.fasta', mode='all_to_all', with_self=False, delete_proteins=params.delete_proteins)

    # Compute ESM embeddings
    # First, check is all embeddings are already computed
    params.model_location = 'esm2_t36_3B_UR50D'
    params.fasta_file = 'sequences.fasta'
    params.output_dir = Path('esm_emb_3B')
    params.include = 'per_tok'

    with open(params.fasta_file, 'r') as f:
        seq_ids = [line.strip().split(' ')[0].replace('>', '') for line in f.readlines() if line.startswith('>')]

    if not os.path.exists(params.output_dir):
        print('Computing ESM embeddings...')
        esm_extract.run(params)
    else:
        for seq_id in seq_ids:
            if not os.path.exists(os.path.join(params.output_dir, seq_id + '.pt')):
                print('Computing ESM embeddings...')
                esm_extract.run(params)
                break

    print('Predicting...')
    main(params)

    os.remove('sequences.fasta')
    os.remove('protein.actions.tsv')
    os.remove('string_interactions.tsv')

    # srun --gres gpu:1 python network_test.py -n 30 --pred_threshold 500 -r 0 -s 9606 -g C1R RFC5 --delete_proteins C1S RFC3 RFC4  DSCC1 CHTF8 RAD17 RPA1 RPA2