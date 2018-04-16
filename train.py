from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from utils import load_data, accuracy
from models import GCN
from sklearn.mixture import GaussianMixture

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--m_steps', type=int, default=5, help='M steps.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--n_communities', type=int, default=20,
                    help='Number of assumed n_communities.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset to use.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
adj, tensor_a, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, n_components=args.n_communities)
var_assignments = Variable(tensor_a.cuda(), requires_grad=True)
# Model and optimizer
model = GCN(dims=[features.shape[1],args.hidden,labels.max() + 1], dropout=args.dropout, use_cuda=args.cuda)
ml = list()
ml.append({'params': model.gc1[0].parameters(), 'weight_decay': args.weight_decay})
ml.append({'params': model.gc1[-1].parameters()})
ml.append({'params': var_assignments})
optimizer = optim.Adam(ml, lr=args.lr)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, labels = Variable(features), Variable(labels)
PI = None
MUs = None
PREs = None

def train():
    global PI, MUs, PREs
    for epoch in range(args.epochs):
        for ite in range(args.m_steps):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output, probs = model(features, adj, PI, MUs, PREs)
            if not probs is None:
                loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + probs*0.01
            else:
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            #gg = torch.autograd.grad(loss_train, var_assignments, only_inputs=True, retain_graph=True)
            #print(var_assignments.grad)

            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output, _ = model(features, adj, PI, MUs, PREs)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch*args.m_steps+ite+1),
                  'loss_train: {:.4f}'.format(loss_train.data[0]),
                  'acc_train: {:.4f}'.format(acc_train.data[0]),
                  'loss_val: {:.4f}'.format(loss_val.data[0]),
                  'acc_val: {:.4f}'.format(acc_val.data[0]),
                  'time: {:.4f}s'.format(time.time() - t))
        model.eval()
        embeddings = model.get_embedding(features, adj).cpu().data.numpy()
        gmm = GaussianMixture(args.n_communities, 'diag').fit(embeddings)
        PI = Variable(torch.Tensor(gmm.weights_).cuda())
        PREs = Variable(torch.Tensor(gmm.precisions_).cuda())
        MUs = Variable(torch.Tensor(gmm.means_).cuda())

def test():
    global PI, MUs, PREs
    model.eval()
    output, _ = model(features, adj, PI, MUs, PREs)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
