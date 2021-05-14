from __future__ import absolute_import, division, print_function

import os
import random
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch

BEAUTY = 'beauty'
CELL = 'cell'
CLOTH = 'clothing'
CD = 'cd'

DATA_DIR = {
    BEAUTY: './data/Beauty',
    CELL: './data/Cellphones_Accessories',
    CLOTH: './data/Clothing',
    CD: './data/CDs_Vinyl',
}

TMP_DIR = {
    BEAUTY: './tmp/Beauty',
    CELL: './tmp/Cellphones_Accessories',
    CLOTH: './tmp/Clothing',
    CD: './tmp/CDs_Vinyl',
}

LABEL_FILE = {
    BEAUTY: (DATA_DIR[BEAUTY] + '/Beauty_train_label.pkl',
             DATA_DIR[BEAUTY] + '/Beauty_test_label.pkl'),
    CELL: (DATA_DIR[CELL] + '/Cellphones_Accessories_train_label.pkl',
           DATA_DIR[CELL] + '/Cellphones_Accessories_test_label.pkl'),
    CLOTH: (DATA_DIR[CLOTH] + '/Clothing_train_label.pkl',
            DATA_DIR[CLOTH] + '/Clothing_test_label.pkl'),
    CD: (DATA_DIR[CD] + '/CDs_Vinyl_train_label.pkl',
         DATA_DIR[CD] + '/CDs_Vinyl_test_label.pkl')
}

EMBED_FILE = {
    BEAUTY: DATA_DIR[BEAUTY] + '/kg_embedding.ckpt',  # '/embedding_des_epoch_29.ckpt'
    CELL: DATA_DIR[CELL] + '/embedding_des_epoch_30.ckpt',
    CLOTH: DATA_DIR[CLOTH] + '/embedding_des_epoch_29.ckpt',
    CD: DATA_DIR[CD] + '/embedding_des_epoch_30.ckpt',
}


# def load_dataset(dataset):
#     dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
#     dataset = pickle.load(open(dataset_file, 'rb'))
#     return dataset


def load_embed_sd(dataset):
    state_dict = torch.load(EMBED_FILE[dataset], map_location=lambda storage, loc: storage)
    return state_dict


def load_embed(dataset):
    embed_file = TMP_DIR[dataset] + '/embed.pkl'
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def save_embed(dataset, embed):
    embed_file = TMP_DIR[dataset] + '/embed.pkl'
    pickle.dump(embed, open(embed_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(embed_file)}".')


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))
    print(f'File is saved to "{os.path.abspath(kg_file)}".')


def load_user_products(dataset, up_type='pos'):
    up_file = '{}/user_products_{}.npy'.format(TMP_DIR[dataset], up_type)
    with open(up_file, 'rb') as f:
        up = np.load(f)
    return up


def save_user_products(dataset, up, up_type='pos'):
    up_file = '{}/user_products_{}.npy'.format(TMP_DIR[dataset], up_type)
    with open(up_file, 'wb') as f:
        np.save(f, up)
    print(f'File is saved to "{os.path.abspath(up_file)}".')

######################################################################################

def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABEL_FILE[dataset][0]
    elif mode == 'test':
        label_file = LABEL_FILE[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products



def load_pos_paths(dataset, mpath_id):
    paths_file = TMP_DIR[dataset] + '/pos_paths_mp{}.pkl'.format(mpath_id)
    paths = pickle.load(open(paths_file, 'rb'))
    return paths


def save_pos_paths(dataset, paths, mpath_id):
    paths_file = TMP_DIR[dataset] + '/pos_paths_mp{}.pkl'.format(mpath_id)
    pickle.dump(paths, open(paths_file, 'wb'))


def load_path_count(dataset):
    count_file = TMP_DIR[dataset] + '/path_count.pkl'
    count = pickle.load(open(count_file, 'rb'))
    return count


def save_path_count(dataset, count):
    count_file = TMP_DIR[dataset] + '/path_count.pkl'
    pickle.dump(count, open(count_file, 'wb'))


def load_mp_split(dataset, ratio, split_type):
    mp_split_file = '{}/mp_split_{:.2f}_{}.pkl'.format(TMP_DIR[dataset], ratio, split_type)
    mp_split = pickle.load(open(mp_split_file, 'rb'))
    return mp_split


def save_mp_split(dataset, mp_split, ratio, split_type):
    mp_split_file = '{}/mp_split_{:.2f}_{}.pkl'.format(TMP_DIR[dataset], ratio, split_type)
    pickle.dump(mp_split, open(mp_split_file, 'wb'))


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
