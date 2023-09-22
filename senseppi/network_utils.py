import json
from Bio import SeqIO
from itertools import permutations
import pandas as pd
import os
import urllib.request
import requests
import gzip
import shutil

DOWNLOAD_LINK_STRING = "https://stringdb-downloads.org/download/"


def generate_pairs_string(fasta_file, output_file, delete_proteins=None):
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)

    pairs = []
    for p in [p for p in permutations(ids, 2)]:
        if (p[1], p[0]) not in pairs and (p[0], p[1]) not in pairs:
            pairs.append(p)

    pairs = pd.DataFrame(pairs, columns=['seq1', 'seq2'])

    data = pd.read_csv('string_interactions.tsv', delimiter='\t')

    # Creating a dictionary of string ids and gene names
    ids_dict = dict(zip(data['preferredName_A'], data['stringId_A']))
    ids_dict.update(dict(zip(data['preferredName_B'], data['stringId_B'])))

    data = data[['stringId_A', 'stringId_B', 'score']]
    data.columns = ['seq1', 'seq2', 'label']

    pairs = pairs.merge(data, on=['seq1', 'seq2'], how='left').fillna(0)

    if delete_proteins is not None:
        print('Labels removed: ', delete_proteins)
        string_ids_to_delete = []
        for label in delete_proteins:
            string_ids_to_delete.append(ids_dict[label])
        print('String ids to delete: ', string_ids_to_delete)
        pairs = pairs[~pairs['seq1'].isin(string_ids_to_delete)]
        pairs = pairs[~pairs['seq2'].isin(string_ids_to_delete)]

    pairs.to_csv(output_file, sep='\t', index=False, header=False)


def get_names_from_string(ids, species):
    string_api_url, _ = get_string_url()
    params = {
        "identifiers": "\r".join(ids),  # your protein list
        "species": species,  # species NCBI identifier
        "limit": 1,  # only one (best) identifier per input protein
        "echo_query": 1,  # see your input identifiers in the output
    }
    request_url = "/".join([string_api_url, "tsv", "get_string_ids"])
    results = requests.post(request_url, data=params)
    lines = results.text.strip().split("\n")
    return pd.DataFrame([line.split('\t') for line in lines[1:]], columns=lines[0].split('\t'))


def get_string_url():
    # Get stable api and current STRING version
    request_url = "/".join(["https://string-db.org/api", "json", "version"])
    response = requests.post(request_url)
    version = json.loads(response.text)[0]['string_version']
    stable_address = json.loads(response.text)[0]['stable_address']
    return "/".join([stable_address, "api"]), version


def get_interactions_from_string(gene_names, species=9606, add_nodes=10, required_score=500, network_type='physical'):
    string_api_url, version = get_string_url()
    output_format = "tsv"
    method = "network"

    # Download protein sequences for given species if not downloaded yet
    if not os.path.isfile('{}.protein.sequences.v{}.fa'.format(species, version)):
        print('Downloading protein sequences')
        url = '{0}protein.sequences.v{1}/{2}.protein.sequences.v{1}.fa.gz'.format(DOWNLOAD_LINK_STRING, version,
                                                                                  species)
        urllib.request.urlretrieve(url, '{}.protein.sequences.v{}.fa.gz'.format(species, version))
        print('Unzipping protein sequences')
        with gzip.open('{}.protein.sequences.v{}.fa.gz'.format(species, version), 'rb') as f_in:
            with open('{}.protein.sequences.v{}.fa'.format(species, version), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove('{}.protein.sequences.v{}.fa.gz'.format(species, version))
        print('Done')

    request_url = "/".join([string_api_url, output_format, method])

    if isinstance(gene_names, str):
        gene_names = [gene_names]

    params = {

        "identifiers": "%0d".join(gene_names),
        "species": species,
        "required_score": required_score,
        "add_nodes": add_nodes,
        "network_type": network_type
    }
    response = requests.post(request_url, data=params)

    lines = response.text.strip().split("\n")
    string_interactions = pd.DataFrame([line.split('\t') for line in lines[1:]], columns=lines[0].split('\t'))

    if 'Error' in string_interactions.columns:
        err_msg = string_interactions['ErrorMessage'].values[0]
        err_msg = err_msg.replace('<br>', '\n').replace('<br/>', '\n').replace('<p>', '\n').replace('</p>', '\n')
        raise Exception(err_msg)
    if len(string_interactions) == 0:
        raise Exception('No interactions found. Please revise your input parameters.')

    # Removing duplicated interactions
    string_interactions.drop_duplicates(inplace=True)
    # Making the interactions symmetric: adding the interactions where the first and second columns are swapped
    string_interactions = pd.concat([string_interactions, string_interactions.rename(
        columns={'stringId_A': 'stringId_B', 'stringId_B': 'stringId_A', 'preferredName_A': 'preferredName_B',
                 'preferredName_B': 'preferredName_A'})])

    string_names_input_genes = get_names_from_string(gene_names, species)
    string_names_input_genes['stringId_A'] = string_names_input_genes['stringId']
    string_names_input_genes['preferredName_A'] = string_names_input_genes['preferredName']
    string_names_input_genes['stringId_B'] = string_names_input_genes['stringId']
    string_names_input_genes['preferredName_B'] = string_names_input_genes['preferredName']
    string_interactions = pd.concat([string_interactions, string_names_input_genes[
        ['stringId_A', 'preferredName_A', 'stringId_B', 'preferredName_B']]])
    string_interactions.fillna(0, inplace=True)

    ids = list(string_interactions['stringId_A'].values) + \
          list(string_interactions['stringId_B'].values) + \
          string_names_input_genes['stringId'].to_list()
    ids = set(ids)

    with open('sequences.fasta', 'w') as f:
        for record in SeqIO.parse('{}.protein.sequences.v{}.fa'.format(species, version), "fasta"):
            if record.id in ids:
                SeqIO.write(record, f, "fasta")
    string_interactions.to_csv('string_interactions.tsv', sep='\t', index=False)


if __name__ == '__main__':
    get_interactions_from_string('RFC5')
