import numpy as np
import scipy.sparse as sp
import torch
import sys
import networkx as nx
import pickle as pkl


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(adj_nonormed):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj_nonormed)
    # adj_n = np.zeros([n_components, adj.shape[0]])
    # for i in range(n_components):
    #     adj_n[i] = (assignments==i).astype(np.float32)
    # p1 = sp.vstack([adj, sp.coo_matrix(adj_n)])
    # p2 = sp.vstack([sp.coo_matrix(adj_n.transpose()), sp.coo_matrix(np.zeros([n_components, n_components]))])
    # adj = sp.hstack([p1, p2])
    # cnt = 0
    # for i in range(1,1+n_components):
    #     tmp = adj.toarray()[-i].sum()
    #     print(tmp)
    #     cnt = cnt + tmp
    # print(cnt)
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sparse_mx_to_torch_sparse_tensor(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo())

def load_data(dataset_str="cora", n_components=10):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    features = normalize(features).toarray()
    # gmm = GaussianMixture(n_components=n_components).fit(features)
    assignments = torch.FloatTensor(np.random.randn(features.shape[0], n_components))
    # assignments = gmm.predict(features)
    # features = np.concatenate([features, normalize(gmm.means_)], 0)
    adj_nonormed = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize_adj(adj_nonormed)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, assignments, features, labels, idx_train, idx_val, idx_test


    # """Load citation network dataset (cora only for now)"""
    # path = 'data/'
    # dataset = dataset_str
    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # #print(adj.toarray()[:10,:10])
    # adj = normalize_adj(adj + adj.T + sp.eye(adj.shape[0]))
    # #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    #
    # return adj, None, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def log_sum_exp(value, dim=1, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=keepdim))
