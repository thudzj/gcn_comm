import numpy as np
import scipy.sparse as sp
import torch
import sys
import networkx as nx
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def encode_onehot(labels, classes=None):
    if classes is None:
        classes = set(labels)
    else:
        classes = range(classes)
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

def normalize_adj(adj_nonormed, use_torch=0):
    """Symmetrically normalize adjacency matrix."""
    if not use_torch:
        adj = sp.coo_matrix(adj_nonormed)
        # adj_n = np.zeros([n_components, adj.shape[0]])
        # for i in range(n_components):
        #     adj_n[i] = (assignments==i).astype(np.float32)
        #assignments = encode_onehot(assignments, n_communities)
        # p1 = sp.hstack([adj + sp.eye(adj.shape[0]), sp.coo_matrix(assignments)])
        # p2 = sp.hstack([sp.coo_matrix(assignments.transpose()), sp.coo_matrix(np.zeros([assignments.shape[1], assignments.shape[1]]))])
        # adj = sp.vstack([p1, p2])
        # cnt = 0
        # for i in range(1,1+n_components):
        #     tmp = adj.toarray()[-i].sum()
        #     print(tmp)
        #     cnt = cnt + tmp
        # print(cnt)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return sparse_mx_to_torch_sparse_tensor(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo())

def load_data(dataset_str="cora", n_communities=20):
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
    kmn = KMeans(n_communities, n_jobs=-1).fit(features)
    means = kmn.cluster_centers_
    #assignments = gmm.predict_proba(features)
    features = np.concatenate((features, normalize(means)))


    adj_nonormed = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #adj = normalize_adj(adj_nonormed, assignments, n_communities)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    idx_test = test_idx_range.tolist()[:-15]
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #adj_nonormed = torch.FloatTensor(sp.coo_matrix().toarray())
    return normalize_adj(adj_nonormed+ sp.eye(labels.shape[0])), features, labels, idx_train, idx_val, idx_test


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
    #print(correct.cpu().data.numpy().mean)
    return correct.mean()


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

def sample_nodes(adj_nonormed, num=100):
    N = adj_nonormed.shape[0]
    flag = np.zeros([N])
    output = [0] * num
    for i in range(num):
        a = np.random.randint(0, N)
        while flag[a] == 1:
            a = np.random.randint(0, N)
        output[i] = a
        flag[a] = 1
        flag[np.nonzero(adj_nonormed[a])] = 1
        for j in np.nonzero(adj_nonormed[a])[0]:
            flag[np.nonzero(adj_nonormed[j])] = 1
        #print(flag.sum())
    output_ = np.ones([N])
    output_[output] = 0
    output_ = np.nonzero(output_)[0]
    return torch.LongTensor(output).cuda(), torch.LongTensor(output_).cuda()

def vat_loss(model, X, adj, adj_nonormed, xi=1e-6, eps=1.0, Ip=1, use_gpu=True):
    """VAT loss function
    :param model: networks to train
    :param X: Variable, input
    :param xi: hyperparameter of VAT (default: 1e-6)
    :param eps: hyperparameter of VAT (default: 1.0)
    :param Ip: iteration times of computing adv noise (default: 1)
    :param use_gpu: use gpu or not (default: True)
    :return: LDS, model prediction (for classification-loss calculation)
    """
    kl_div = nn.KLDivLoss(reduce=False)
    if use_gpu:
        kl_div.cuda()


    pred = model(X, adj)
    index_adv, index_ori = sample_nodes(adj_nonormed, 300)

    # prepare random unit tensor
    d = torch.rand(X.shape)
    d = Variable(F.normalize(d, p=2, dim=1).cuda(), requires_grad = True)
    # calc adversarial direction
    for ip in range(Ip):
        d= Variable(d.data * xi, requires_grad = True)
        pred_hat = model((X + d), adj)
        adv_distance = kl_div(F.log_softmax(pred_hat, dim=1), F.softmax(pred.detach())).sum(1)[index_adv].mean()
        adv_distance.backward()
        d = Variable(F.normalize(d.grad.data, p=2, dim=1).cuda(), requires_grad = True)
        model.zero_grad()

    # calc LDS
    # model.train()
    # pred = model(X, adj)

    r_adv = d * eps
    # r_adv[index_ori] = 0
    X_hat = X + r_adv
    #X_hat = X_hat / X_hat.sum(1, keepdim=True)
    pred_hat = model(X_hat, adj)
    print(accuracy(pred_hat[index_adv], pred[index_adv].max(1)[1]))
    # model.train()
    # pred = model(X, adj)
    LDS = kl_div(F.log_softmax(pred_hat, dim=1), F.softmax(pred.detach())).sum(1)[index_adv].mean()

    return LDS - torch.sum(F.log_softmax(pred, dim=1)*F.softmax(pred, dim=1), 1).mean()
