import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
import pandas as pd
import math
import random
import os
from torch import nn

import itertools

from sklearn.metrics import f1_score as lib_f1_score
from sklearn.metrics import roc_auc_score as lib_roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
# from sklearn.metrics import roc_curve as lib_roc_curve

def flatten(list2d):
    return list(itertools.chain.from_iterable(list2d))

def slicing(data, index_ranges):
    '''
    index_ranges in the form of a list of [begin, end] pairs
    '''
    slices = list()
    for range_ in index_ranges:
        slices.append(data[range_[0]: range_[1]])
    return np.concatenate(slices)

def pair_set(from_list, to_list, reverse=True):
    pairs = set(zip(from_list,to_list))
    if reverse:
        return pairs.union(zip(to_list,from_list))
    else:
        return pairs

def from_list(tuple_list):
    from_list = [elem[0] for elem in tuple_list]
    return from_list

def to_list(tuple_list):
    to_list = [elem[1] for elem in tuple_list]
    return to_list

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# this is to calculate the normalized Laplacian
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1_score(output, labels, average='macro'):
    '''
    average could be macro, micro, weighted
    '''
    preds = output.max(1)[1].type_as(labels)
    return lib_f1_score(labels, preds, average=average)

def roc_auc_score(all_scores, all_labels):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    '''
    weights = list()
    results = list()
    for r,(scores,labels) in enumerate(zip(all_scores, all_labels)):
        weights.append(len(scores))
        results.append(lib_roc_auc_score(labels, scores))
        # equivalent expression
        #fpr, tpr, thresholds = roc_curve(labels, scores)
        #results.append(auc(fpr, tpr))
    return np.average(results, weights=weights), results

def precision_recall_score(all_scores, all_labels):
    '''
    precision-recall curve
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    other metrics:
    https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    '''
    weights = list()
    results = list()
    for r,(scores,labels) in enumerate(zip(all_scores, all_labels)):
        weights.append(len(scores))
        results.append(average_precision_score(labels, scores))
        # results.append(average_precision_score(labels, scores))
        # precision, recall, thresholds = precision_recall_curve(labels, scores)
        # results.append(auc(recall, precision))
    return np.average(results, weights=weights), results

def link_accuracy(all_scores, all_labels, min_step_digit=2):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    '''
    weights = list()
    results = list()
    thresholds = list()
    min_step = 10 ** (-min_step_digit)
    for r,(scores,labels) in enumerate(zip(all_scores, all_labels)):
        len_data = len(scores)
        weights.append(len_data)
        # there'll be too many thresholds to test on in this way
        # fpr, tpr, thresholds = lib_roc_curve(labels, scores, pos_label=1)
        sorted_list = sorted(zip(scores, labels), key=lambda x: x[0])
        current_threshold = round(sorted_list[0][0], min_step_digit)
        current_correct = sum(labels)
        max_correct = 0
        best_threshold = -1
        # >= threshold: positive prediction
        idx = 0
        if current_threshold > sorted_list[0][0]:
            if sorted_list[0][1] == 1:
                current_correct -= 1
            else:
                current_correct += 1
            idx = 1
        best_threshold = current_threshold
        max_correct = current_correct
        while current_threshold < sorted_list[-1][0]:
            # not-yet reached the next data point
            if current_threshold <= sorted_list[idx][0]: 
                current_threshold += min_step
                continue
            # reached the next data point
            if sorted_list[idx][1] == 1:
                current_correct -= 1
            else:
                current_correct += 1
                if current_correct > max_correct:
                    max_correct = current_correct
                    best_threshold = current_threshold
            idx += 1
        results.append(max_correct / len_data)
        thresholds.append(best_threshold)
    return np.average(results, weights=weights), thresholds

def link_accuracy_fixed_thresholds(all_scores, all_labels, thresholds):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    '''
    weights = list()
    results = list()
    for r,(scores,labels,threshold) in enumerate(zip(all_scores, all_labels, thresholds)):
        len_data = len(scores)
        weights.append(len_data)
        sorted_list = sorted(zip(scores, labels), key=lambda x: x[0])
        over_threshold = False
        correct = 0
        for score,label in sorted_list:
            if over_threshold:
                if label == 1: correct += 1
            else:
                if score >= threshold:
                    over_threshold = True
                    if label == 1: correct += 1
                else:
                    if label == 0: correct += 1
        results.append(correct / len_data)
    return np.average(results, weights=weights)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo(sparse_mx.shape).astype(np.float32)
    # the above line could be tricky and cause problems at times because of memory issue
    # but that's convenient, as it makes it possible for us to use
    #     sparse_mx.row, sparse_mx.col
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def calculate_sym_laplacian(adj):
    total_nodes = adj.shape[0]
    degree = adj.sum(axis=1)
    degree_sqrt_inv = np.sqrt(1.0 / degree).reshape(1, -1)
    degree_sqrt_inv[np.isinf(degree_sqrt_inv)] = 0.
    D_sqrt_inv = sp.diags(degree_sqrt_inv, [0], shape=(total_nodes, total_nodes))
    adj_tilde = np.dot(np.dot(D_sqrt_inv, adj), D_sqrt_inv)
    return adj_tilde

# our own implementation calculating non-normalized laplacian
def calculate_laplacian(adj, orientation='row'):
    if orientation is None:
        # that is a Laplacian
        return normalize(adj)
    total_nodes = adj.shape[0]
    degree = adj.sum(axis=1) + 1 # + 1 for smoothing
    degree_inverse = (1.0/degree).reshape(1, -1)
    degree_inverse[np.isinf(degree_inverse)] = 0.
    D_inverse = sp.diags(degree_inverse, [0], shape=(total_nodes, total_nodes))
    adj_tilda = 0
    if orientation == 'row':
        adj_tilda = D_inverse @ adj
    else:
        adj_tilda = adj @ D_inverse
    return adj_tilda

def index_to_raw_edges_info(from_to_counts, indexes):
    from_idx_raw, to_idx_raw, counts_raw = from_to_counts
    return np.array([from_idx_raw[indexes], to_idx_raw[indexes], counts_raw[indexes]])

def generate_adjs_masks(mask_info, n_entities):
    from_info, to_info = mask_info
    all_masks = list()
    for from_idx, to_idx in zip(from_info, to_info):
        n_links = len(from_idx)
        tmp_mask = sp.csr_matrix((np.ones(n_links), (from_idx, to_idx)),
                         shape=(n_entities, n_entities),
                         dtype=np.float32)
        tmp_mask_reverse = sp.csr_matrix((np.ones(n_links), (to_idx, from_idx)),
                         shape=(n_entities, n_entities),
                         dtype=np.float32)
        all_masks.append(sparse_mx_to_torch_sparse_tensor(tmp_mask))
        all_masks.append(sparse_mx_to_torch_sparse_tensor(tmp_mask_reverse))
    all_masks.append(sparse_mx_to_torch_sparse_tensor(sp.eye(n_entities)))
    return all_masks

def apply_masks(adjs, masks):
    return [adj - adj * mask for adj,mask in zip(adjs, masks)]

def debug_mask():
    '''
    the function used to debug the masks...
    NOT really needed in actual implementation
    '''
    n_entities = 5
    rows = list()
    for i in range(n_entities):
        rows.extend([i] * n_entities)
    adj = sparse_mx_to_torch_sparse_tensor(
                sp.csr_matrix((np.ones(n_entities * n_entities) * 0.5, (rows, list(range(n_entities)) * n_entities)),
                         shape=(n_entities, n_entities),
                         dtype=np.float32))
    print(adj.to_dense())
    mask_info = ([[1, 2, 3, 4, 0]],[[0,1,2,3,4]])
    print(list(zip(mask_info[0][0], mask_info[1][0])))
    print(generate_masked_adjs(mask_info, n_entities, [adj])[0].to_dense())
    exit(0)

def generate_masked_adjs(mask_info, n_entities, adjs):
    all_masks = generate_adjs_masks(mask_info, n_entities)
    masked_adjs = apply_masks(adjs, all_masks)
    return masked_adjs

##################################################################
# The data loader that loads the data as required
#    such that later on we can pass them into the MTL model
#    (half-way processed)
##################################################################
def multi_relation_load(path="../data/PureP", label="dict.csv",
                        files=["friend_list.csv", "retweet_list.csv"],
                        label_key = "twitter_id", label_property = "party", ignored_labels = ["I"],
                        calc_lap=None, separate_directions=True, feature_data="one_hot", feature_file=None, feat_order_file="all_twitter_ids.csv",
                        split_links=False, portion={"valid": 0.05, "test": 0.1}, freeze_feature=False,
                        additional_labels_files=["../data/additional_labels/new_dict_cleaned.csv"], add_additional_labels=True):
    print("Loading data from path {0}".format(path))
    print("  relations: {}".format(" ".join(files)))
    assert calc_lap in ["col", "row", None], "calc_lap must be row, column or None"
    DATA = Path(path)
    FILE = [DATA/i for i in files]
    LABEL = DATA/label
    data_dfs = []
    label_df = pd.read_csv(LABEL, sep="\t")
    for file in FILE:
        data_dfs.append(pd.read_csv(file, sep="\t"))
    # get the valid id list
    all_ids = set()
    for df in data_dfs:
        all_ids = all_ids.union(set(df[df.columns[0]]))
        all_ids = all_ids.union(set(df[df.columns[1]]))

    labeled_ids = label_df[label_key].values
    label_list = label_df[label_property].values

    # decide if we would like to use additional labels
    if add_additional_labels:
        print("\tloading additional labels from {} files: {}".format(len(additional_labels_files), ", ".join(additional_labels_files) if len(additional_labels_files) else "None"))
        additional_labels_dfs = [pd.read_csv(fname, sep="\t") for fname in additional_labels_files]
        for tmp_df in additional_labels_dfs:
            # if there are additional labels
            labeled_ids = np.append(labeled_ids, tmp_df[label_key].values)
            label_list = np.append(label_list, tmp_df[label_property].values)

    # this might be a very low-efficient implementation, but in our case it is okay
    #     because our labeled entities are too sparse, we don't bother optimize this part
    ignored_idxes = [i for i,(tid,v) in enumerate(zip(labeled_ids,label_list)) if v in ignored_labels or tid not in all_ids]
    # ignored_ids = labeled_ids[ignored_idxes]
    # remove the ignored parts
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.delete.html
    labeled_ids= np.delete(labeled_ids, ignored_idxes)
    label_list = np.delete(label_list, ignored_idxes)
    # more general use: sort in alphabet order
    # in D, R case: we remove I (too few) then it is D = 0 and R = 1
    # D for democratics, I for Independent, R for Republican
    label_categories = list(set(label_list))                                # expected to be ['D', 'R']
    label_categories.sort()
    label_map = dict(zip(label_categories, range(len(label_categories))))
    labels = list(map(label_map.get, label_list))
    n_labels = len(labels)                          # portion .6:.2:.2 if there's no additional label
    n_train = math.ceil(n_labels * .8) 
    n_valid = math.ceil(n_labels * .1) 
    idx_all = list(range(n_labels))

    random.shuffle(idx_all)                         # permutation added
    idx_train = idx_all[:n_train]
    idx_val = idx_all[n_train: n_train + n_valid]
    idx_test = idx_all[n_train + n_valid: ]
    
    print("\tprocessing nodes")
    
    unlabeled_ids = all_ids - set(labeled_ids)
    all_id_list = np.concatenate((  np.array(labeled_ids, dtype=np.int64), 
                                    np.array(list(unlabeled_ids), dtype=np.int64)
                                ))
    n_entities = len(all_id_list)
    idx_map = {j: i for i, j in enumerate(all_id_list)}
    
    print("\tprocessing edges")

    adjs = list()

    triplets = None
    relations = [f_name.split('_')[0] for f_name in files]
    idx_relation = 0

    train_link_info = list()
    valid_link_info = list()
    test_link_info = list()
    # make sure that the portion makes sense
    if split_links:
        test_split_ratio = portion["test"]
        full_split_ratio = portion["test"] + portion["valid"] 
        assert full_split_ratio < 1, "validation set and test set takes up to 100%"
    
    for data_df in data_dfs:
        n_edges = len(data_df)
        # we can do positive sampling of the edges here, in link prediction
        from_idx_raw = np.array(list(map(idx_map.get, data_df[data_df.columns[0]].values)), dtype=np.int64)
        to_idx_raw = np.array(list(map(idx_map.get, data_df[data_df.columns[1]].values)), dtype=np.int64)
        counts_raw = np.array(data_df[data_df.columns[2]].values, dtype=np.int64)

        # valid_adj_info.append((from_idx,to_idx,counts))
        if not split_links:
            from_idx = from_idx_raw
            to_idx = to_idx_raw
            counts = counts_raw
        else:
            # do splitting by dividing the indexes
            test_split_end = math.ceil(n_edges * test_split_ratio)
            full_split_end = math.ceil(n_edges * full_split_ratio)
            all_edges_index = list(range(n_edges))
            random.shuffle(all_edges_index)
            test_split_index = all_edges_index[:test_split_end]
            valid_split_index = all_edges_index[test_split_end:full_split_end]
            train_split_index = all_edges_index[full_split_end:]

            raw_data = (from_idx_raw, to_idx_raw, counts_raw)

            from_idx, to_idx, counts = index_to_raw_edges_info(raw_data, train_split_index)
            train_link_info.append(np.array([from_idx, to_idx, counts]))
            valid_link_info.append(index_to_raw_edges_info(raw_data, valid_split_index))
            test_link_info.append(index_to_raw_edges_info(raw_data, test_split_index))

        adj = sp.csr_matrix((counts, (from_idx, to_idx)),
                        shape=(n_entities, n_entities),
                        dtype=np.float32)

        # if build symmetric adjacency matrix, in this case we'll have r adjacancy matrix in the end
        if not separate_directions:
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adjs.append(calculate_laplacian(adj, calc_lap))
        else:
            # otherwise, we'll have 2r + 1 adjacency matrix
            adjs.append(calculate_laplacian(adj, calc_lap))
            adjs.append(calculate_laplacian(adj.T, calc_lap))
            
    
    edge_indexs = np.array(range(n_entities))

    if separate_directions:
        self_loop = sp.csr_matrix((np.ones(n_entities), (edge_indexs, edge_indexs)), 
                                  shape=(n_entities, n_entities), dtype=np.float32)
        adjs.append(calculate_laplacian(self_loop, calc_lap))

    print("\tprocessing features")

    trainable = None
    mask = None
    
    if feature_data == "one_hot":
        # one-hot
        features = sp.eye(n_entities)
        features = normalize(features)
        # transfering into tensors
        features_ebm = sparse_mx_to_torch_sparse_tensor(features)
    elif feature_data is "random":
        # randomnized
        features = sp.random(n_entities, 300, density=1.) # density is optional
        features = normalize(features)
        # transfering into tensors
        features_ebm = sparse_mx_to_torch_sparse_tensor(features)
    elif type(feature_data) == str:
        feature_path = os.path.join(path, feature_file)
        features_unordered = np.load(feature_path)[feature_data]
        features_unordered = np.vstack((features_unordered, np.zeros(features_unordered.shape[1]))) # the last position is used for unseen node ids
        twitter_id_ordered = list(pd.read_csv(os.path.join(path, feat_order_file), sep="\t")["twitter_id"])
        # twitter_id_current = all_id_list
        missing_feature = len(features_unordered) - 1
        tid2fidx = dict(zip(twitter_id_ordered, range(len(twitter_id_ordered)))) # feature list index
        features_idx_list = list(map(lambda k: tid2fidx.get(k, missing_feature), all_id_list))
        features = features_unordered[features_idx_list]
        trainable_idx = np.where(~features.any(axis=1))[0]
        features = normalize(features)
        # make a mask
        mask = np.zeros(features.shape[0])
        mask[trainable_idx] = 1.
        # create the embeddings we need
        features_ebm = nn.Embedding.from_pretrained(torch.FloatTensor(features))
        trainable = nn.Embedding(features.shape[0], features.shape[1])
        trainable.weight.requires_grad = not freeze_feature
        # turn mask into tensors
        mask = torch.LongTensor(mask)
    else:
        print("undefined feature type {}".format(type(feature_data)))
        exit(0)
    
    labels = torch.LongTensor(labels)
    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adjs, features_ebm, (idx_train, idx_val, idx_test, labels), trainable, mask, (train_link_info, valid_link_info, test_link_info), (label_map, all_id_list)

def save_node_pred(classifier_out_numpy, file_name, label_map, all_id_list, file_dir="../results/", task=""):
    # file_path = os.path.join(file_dir, file_name, "node.csv") # file_dir="../data/"
    file_path = os.path.join(file_dir, "_".join([task,file_name,"node.csv"]))
    node_pred = np.argmax(classifier_out_numpy, axis=1)
    reversed_label_map = dict(zip(label_map.values(), label_map.keys()))
    node_pred_label = list(map(reversed_label_map.get, node_pred))

    #from scipy.special import softmax
    #node_pred2 = np.argmax(softmax(classifier_out_numpy, axis=1), axis=1)
    #print(node_pred == node_pred2)

    data=pd.DataFrame(data={"twitter_id": all_id_list, "party": node_pred_label})
    data.to_csv(file_path, index=None)

def save_link_pred(link_pred_info, file_name, relations, all_id_list, file_dir="../results/", task=""):
    (all_scores, all_labels, all_from, all_to) = link_pred_info
    file_paths = [os.path.join(file_dir, "_".join([task,file_name,r,"link.csv"])) for r in relations]
    idx2id = dict(zip(range(len(all_id_list)), all_id_list))
    for file_path,score,label,from_,to_ in zip(file_paths, all_scores, all_labels, all_from, all_to):
        from_id = list(map(idx2id.get, from_))
        to_id = list(map(idx2id.get, to_))
        data=pd.DataFrame(data={"twitter_id_from": from_id, "twitter_id_to": to_id, "score": score, "label": label})
        data.to_csv(file_path, index=None)









