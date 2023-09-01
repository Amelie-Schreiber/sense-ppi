import pandas as pd
import os
from tqdm import tqdm
from Bio import SeqIO
import logging
import argparse
import subprocess
from urllib.error import HTTPError
import wget
import gzip
import shutil
import random
from ..network_utils import get_string_url, DOWNLOAD_LINK_STRING


def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


# A class containing methods for preprocessing the full STRING data
class STRINGDatasetCreation:
    def __init__(self, params):
        self.interactions_file = params.interactions
        self.sequences_file = params.sequences
        self.min_length = params.min_length
        self.max_length = params.max_length
        self.species = params.species
        self.max_positive_pairs = params.max_positive_pairs
        self.combined_score = params.combined_score
        self.experimental_score = params.experimental_score

        self.intermediate_file = 'interactions_intermediate.tmp'
        self.proprocced_protein_set = None

        if not params.not_remove_long_short_proteins:
            self.process_fasta_file()

        self._select_interactions()

    def _select_interactions(self):
        # Creating a new intermediate file with only the interactions that are suitable for training
        # Such interaction happen only between proteins of appropriate lenghs
        #
        # And either have a high combined score of > 700
        #
        # Further on, redundant interactions are removed as well as sequences with inappropriate
        # length and interactions based on homology

        if not os.path.isfile(self.intermediate_file):
            if not os.path.isfile('clusters.tsv'):
                logging.info('Running mmseqs to clusterize proteins')
                logging.info('This might take a while if you secide to process the whole STRING database.')
                logging.info(
                    'In order to install mmseqs (if not installed), please visit https://github.com/soedinglab/MMseqs2')
                commands = "; ".join(["mkdir mmseqDBs",
                                      "mmseqs createdb {} mmseqDBs/DB".format(self.sequences_file),
                                      "mmseqs cluster mmseqDBs/DB mmseqDBs/clusterDB tmp --min-seq-id 0.4 --alignment-mode 3 --cov-mode 1 --threads 8",
                                      "mmseqs createtsv mmseqDBs/DB mmseqDBs/DB mmseqDBs/clusterDB clusters.tsv",
                                      "rm -r mmseqDBs",
                                      "rm -r tmp"])
                ret = subprocess.run(commands, shell=True, capture_output=True)
                print(ret.stdout.decode())
                print(ret.stderr.decode())
                logging.info('Clusters file created')

            # Compute the length of self.interactions_file
            with open(self.interactions_file, 'rb') as f:
                c_generator = _count_generator(f.raw.read)
                n_interactions = sum(buffer.count(b'\n') for buffer in c_generator) + 1

            logging.info('Removing redundant interactions')

            clusters = pd.read_csv('clusters.tsv', sep='\t', header=None,
                                   names=['cluster', 'protein']).set_index('protein')
            clusters = clusters.to_dict()['cluster']
            logging.info('Clusters file loaded')

            print('Proteins in clusters: {}'.format(len(clusters)))

            existing_cluster_pairs = set()

            infomsg = 'Extracting only entries with no homology and:\n' \
                      'combined score >= {} '.format(self.combined_score)
            if self.experimental_score is not None:
                infomsg += '\nexperimental score >= {}'.format(self.experimental_score)
            logging.info(infomsg)

            with open(self.intermediate_file, 'w') as f:

                with open(self.interactions_file, 'r') as f2:

                    f.write('\t'.join(f2.readline().strip().split(' ')) + '\n')

                    for line in tqdm(f2, total=n_interactions):
                        line = line.strip().split(' ')

                        if self.species is not None:
                            if not line[0].startswith(self.species) or not line[1].startswith(self.species):
                                continue

                        if self.experimental_score is not None and int(line[3]) < self.experimental_score:
                            continue

                        if int(line[2]) == 0 and int(line[-1]) >= self.combined_score:
                            try:
                                cluster1 = clusters[line[0]]
                                cluster2 = clusters[line[1]]
                            except KeyError:
                                continue

                            if cluster1 < cluster2:
                                cluster_pair = (cluster1, cluster2)
                            else:
                                cluster_pair = (cluster2, cluster1)
                            if cluster_pair not in existing_cluster_pairs:
                                existing_cluster_pairs.add(cluster_pair)
                                f.write('\t'.join(line) + '\n')

            print('Number of proteins: {}'.format(len(clusters)))
            print('Number of interactions: {}'.format(len(existing_cluster_pairs)))

        print('Intermediate preprocessing done.')
        print('You can find the preprocessed file in {}. If the dataset creation is fully done, '
              'it will be deleted.'.format(self.intermediate_file))

    def final_preprocessing_positives(self):
        # This function generates the final preprocessed file.
        data = pd.read_csv(self.intermediate_file, sep='\t')

        # Here you can put further constraints on the interactions.
        # For example, you can further remove unreliable interactions.

        data = data[['protein1', 'protein2', 'combined_score']]

        if self.max_positive_pairs is not None:
            self.max_positive_pairs = min(self.max_positive_pairs, len(data))
            data = data.sort_values(by=['combined_score'], ascending=False).iloc[:self.max_positive_pairs]

        data['combined_score'] = 1

        data.to_csv("protein.pairs_{}.tsv.tmp".format(self.species), sep='\t', index=False)

        # Create new fasta file with only the proteins that are in the interactions file
        proteins = set(data['protein1'].unique()).union(set(data['protein2'].unique()))
        self.proprocced_protein_set = proteins
        with open("sequences_{}.fasta".format(self.species), 'w') as f:
            for record in tqdm(SeqIO.parse(self.sequences_file, "fasta")):
                if record.id in proteins:
                    SeqIO.write(record, f, "fasta")

        logging.info('Final preprocessing for only positive pairs done.')

    def create_negatives(self):
        if not os.path.isfile('clusters_preprocessed.tsv'):
            logging.info(
                'Running mmseqs to compute pairwise sequence similarity for all proteins in preprocceced file.')
            logging.info('This might take a while.')
            commands = "; ".join(["mkdir mmseqDBs",
                                  "mmseqs createdb sequences_{}.fasta mmseqDBs/DB".format(self.species),
                                  "mmseqs cluster mmseqDBs/DB mmseqDBs/clusterDB tmp --min-seq-id 0.4 --alignment-mode 3 --cov-mode 1 --threads 8",
                                  "mmseqs createtsv mmseqDBs/DB mmseqDBs/DB mmseqDBs/clusterDB clusters_preprocessed.tsv",
                                  "rm -r mmseqDBs",
                                  "rm -r tmp"])
            ret = subprocess.run(commands, shell=True, capture_output=True)
            print(ret.stdout.decode())
            print(ret.stderr.decode())
            logging.info('Clusters file created')

        clusters_preprocessed = pd.read_csv('clusters_preprocessed.tsv', sep='\t', header=None,
                                            names=['cluster', 'protein']).set_index('protein')
        clusters_preprocessed = clusters_preprocessed.to_dict()['cluster']

        # Creating new protein.pairs.tsv file that will be used for training This file will contain both positive and
        # negative pairs with ratio 1:10 The negative pairs will be generated using the clusters file: making sure
        # that any paired protein is not in the same cluster with proteins interacting with a given one already.
        # This is done to make sure that the negative pairs are not too similar to the positive ones

        proteins = list(clusters_preprocessed.keys())
        interactions = pd.read_csv("protein.pairs_{}.tsv.tmp".format(self.species), sep='\t')

        logging.info('Generating negative pairs.')
        tqdm.pandas()

        proteins1 = random.choices(proteins, k=len(interactions) * 12)
        proteins2 = random.choices(proteins, k=len(interactions) * 12)
        negative_pairs = pd.DataFrame({'protein1': proteins1, 'protein2': proteins2, 'combined_score': 0})

        logging.info('Negative pairs generated. Filtering out duplicates.')

        # Make protein1 and protein2 in alphabetical order
        negative_pairs['protein1'], negative_pairs['protein2'] = zip(*negative_pairs.progress_apply(
            lambda x: (x['protein1'], x['protein2']) if x['protein1'] < x['protein2'] else (
                x['protein2'], x['protein1']), axis=1))
        negative_pairs = negative_pairs.drop_duplicates()

        logging.info('Duplicates filtered out. Filtering out pairs that are already in the positive interactions file.')

        negative_pairs = negative_pairs[
            ~negative_pairs.progress_apply(lambda x: len(interactions[(interactions['protein1'] == x[
                'protein1']) & (interactions['protein2'] == x['protein2'])]) > 0, axis=1)]
        negative_pairs = negative_pairs[
            ~negative_pairs.progress_apply(lambda x: len(interactions[(interactions['protein1'] == x[
                'protein2']) & (interactions['protein2'] == x['protein1'])]) > 0, axis=1)]

        logging.info(
            'Pairs that are already in the positive interactions file filtered out. Filtering out pairs that are in '
            'the same cluster with proteins interacting with a given one already.')

        negative_pairs = negative_pairs[~negative_pairs.progress_apply(
            lambda x: clusters_preprocessed[x['protein2']] in [clusters_preprocessed[i] for i in
                                                               interactions[interactions['protein1'] == x['protein1']][
                                                                   'protein2'].unique()], axis=1)]

        assert len(negative_pairs) > len(interactions) * 10, 'Not enough negative pairs generated. P' \
                                                             'lease try again and increase the number of pairs (>1000).'

        negative_pairs = negative_pairs.iloc[:len(interactions) * 10]

        logging.info('Negative pairs generated. Saving to file.')

        interactions = pd.concat([interactions, negative_pairs], ignore_index=True)
        interactions.to_csv(os.path.join("protein.pairs_{}.tsv".format(self.species)), sep='\t',
                            index=False,
                            header=False)

        os.remove("protein.pairs_{}.tsv.tmp".format(self.species))
        os.remove(self.intermediate_file)
        os.remove("clusters_preprocessed.tsv")
        os.remove("clusters.tsv")

    # A method to remove sequences of inappropriate length from a fasta file
    def process_fasta_file(self):
        logging.info('Getting protein names out of fasta file.')
        logging.info(
            'Removing proteins that are shorter than {}aa or longer than {}aa.'.format(self.min_length,
                                                                                       self.max_length))
        with open('seqs.tmp', 'w') as f:
            for record in tqdm(SeqIO.parse(self.sequences_file, "fasta")):
                if len(record.seq) < self.min_length or len(record.seq) > self.max_length:
                    continue
                record.description = ''
                record.name = ''
                SeqIO.write(record, f, "fasta")
        # Rename the temporary file to the original file
        os.rename('seqs.tmp', self.sequences_file)


def add_args(parser):
    parser.add_argument("species", type=str,
                        help="The Taxon identifier of the organism of interest.")
    parser.add_argument("--interactions", type=str, default=None,
                        help="The physical links (full) file from STRING for the "
                             "organism of interest.")
    parser.add_argument("--sequences", type=str, default=None,
                        help="The sequences file downloaded from the same page of STRING. "
                             "For both files see https://string-db.org/cgi/download")
    parser.add_argument("--not_remove_long_short_proteins", action='store_true',
                        help="Whether to remove proteins that are too short or too long. "
                             "Normally, the long and short proteins are removed.")
    parser.add_argument("--min_length", type=int, default=50,
                        help="The minimum length of a protein to be included in the dataset.")
    parser.add_argument("--max_length", type=int, default=800,
                        help="The maximum length of a protein to be included in the dataset.")
    parser.add_argument("--max_positive_pairs", type=int, default=None,
                        help="The maximum number of positive pairs to be included in the dataset. "
                             "If None, all pairs are included. If specified, the pairs are selected "
                             "based on the combined score in STRING.")
    parser.add_argument("--combined_score", type=int, default=500,
                        help="The combined score threshold for the pairs extracted from STRING. "
                             "Ranges from 0 to 1000.")
    parser.add_argument("--experimental_score", type=int, default=None,
                        help="The experimental score threshold for the pairs extracted from STRING. "
                             "Ranges from 0 to 1000. Default is None, which means that the experimental "
                             "score is not used.")

    return parser


def main(params):
    downloaded_flag = False
    if params.interactions is None or params.sequences is None:
        downloaded_flag = True
        logging.info('One or both of the files are not specified (interactions or sequences). '
                     'Downloading from STRING...')

        _, version = get_string_url()
        logging.info('STRING version: {}'.format(version))

        try:
            url = "{0}protein.physical.links.full.v{1}/{2}.protein.physical.links.full.v{1}.txt.gz".format(DOWNLOAD_LINK_STRING, version, params.species)
            string_file_name_links = "{1}.protein.physical.links.full.v{0}.txt".format(version, params.species)
            wget.download(url, out=string_file_name_links+'.gz')
            with gzip.open(string_file_name_links+'.gz', 'rb') as f_in:
                with open(string_file_name_links, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            url = "{0}protein.sequences.v{1}/{2}.protein.sequences.v{1}.fa.gz".format(DOWNLOAD_LINK_STRING, version, params.species)
            string_file_name_seqs = "{1}.protein.sequences.v{0}.fa".format(version, params.species)
            wget.download(url, out=string_file_name_seqs+'.gz')
            with gzip.open(string_file_name_seqs+'.gz', 'rb') as f_in:
                with open(string_file_name_seqs, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except HTTPError:
            raise Exception('The files are not available for the specified species. '
                         'There might be two reasons for that: \n '
                         '1) the species is not available in STRING. Please check the STRING species list to verify. \n'
                         '2) the download link has changed. Please raise an issue in the repository. ')

        os.remove(string_file_name_seqs+'.gz')
        os.remove(string_file_name_links+'.gz')

        params.interactions = string_file_name_links
        params.sequences = string_file_name_seqs

    data = STRINGDatasetCreation(params)

    data.final_preprocessing_positives()
    data.create_negatives()

    if downloaded_flag:
        os.remove(params.interactions)
        os.remove(params.sequences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    params = parser.parse_args()
    main(params)

