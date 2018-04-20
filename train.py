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

from utils import load_data, accuracy, vat_loss
from models import GCN
from sklearn.mixture import GaussianMixture

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--n_communities', type=int, default=20,
                    help='Number communities.')
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
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.n_communities)

# Model and optimizer
model = GCN(dims=[features.shape[1]] + [args.hidden]*1 + [labels.max() + 1], hidden_atten=100 ,dropout=args.dropout, use_cuda=args.cuda)
# ml = list()
# ml.append({'params': model.gc1[0].parameters(), 'weight_decay': args.weight_decay})
# ml.append({'params': model.gc1[1].parameters()})
# ml.append({'params': model.attns[1].parameters()})
# ml.append({'params': model.attns[0].parameters()})
# optimizer = optim.Adam(ml, lr=args.lr)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    adj = adj.cuda()

features, labels = Variable(features), Variable(labels)
def train():
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        output = model(features, adj, args.n_communities)
        loss_train = F.nll_loss(F.log_softmax(output[idx_train], dim=1), labels[idx_train])# + 1.3*vat_loss(model, features, adj, adj_nonormed, eps=0.03)
        #gg = torch.autograd.grad(loss_train, var_assignments, only_inputs=True, retain_graph=True)
        #print(var_assignments.grad)

        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj, args.n_communities)

        loss_val = F.nll_loss(F.log_softmax(output[idx_val], dim=1), labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data[0]),
              'acc_train: {:.4f}'.format(acc_train.data[0]),
              'loss_val: {:.4f}'.format(loss_val.data[0]),
              'acc_val: {:.4f}'.format(acc_val.data[0]),
              'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj, args.n_communities)
    test_output = torch.cat((output[idx_test], output[-15:]))
    test_label = torch.cat((labels[idx_test],labels[-15:])) #torch.index_select(labels, 0, Variable(idx_test))
    loss_test = F.nll_loss(F.log_softmax(test_output,dim=1), test_label)
    acc_test = accuracy(test_output, test_label)
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
