import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Attention
import torch
from torch.autograd.variable import Variable
import numpy as np
from utils import normalize_adj, log_sum_exp



class GCN(nn.Module):
    def __init__(self, dims, hidden_atten, dropout, use_cuda):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(dims[0], dims[1], use_cuda)
        self.gc2 = GraphConvolution(dims[1], dims[2], use_cuda)
        self.attn1 = Attention(dims[0], hidden_atten, use_cuda)
        self.attn2 = Attention(dims[1], hidden_atten, use_cuda)
        self.dropout = dropout
        self.use_cuda = use_cuda

    def get_embedding(self, x, adj):
        o = self.gc2[0](self.gc1[0](F.dropout(x, self.dropout, training=self.training)), adj)
        return o

    def forward(self, x, adj, n_communities):
        ## update comms
        # assignments = F.softmax(var_assignments)
        # assignments_t = assignments.t()/assignments.t().sum(1, keepdim=True)#F.softmax(var_assignments.t())
        # x_comms = torch.mm(assignments_t, x)
        #
        # x_comms = self.gc1[0](F.dropout(x_comms, self.dropout, training=self.training))
        assignments = self.attn1(x[:-n_communities], x[-n_communities:])
        x = self.gc1(F.dropout(x, self.dropout, training=self.training), adj, assignments)
        # if not PI is None:
        #     probs = -torch.max(-0.5 * torch.sum((x.unsqueeze(1) - MUs.unsqueeze(0))**2 * PREs.unsqueeze(0), 2) + torch.log(PI) + 0.5*torch.log(PREs).sum(1)-8.0*np.log(2*np.pi))[0].mean()
        #     #probs = F.nll_loss(F.log_softmax(probs), probs.max(1)[1])#-( * F.softmax(probs)).sum(1).mean()
        # else:
        #     probs = None
        # x_comms1 = F.relu(0.5*x_comms + 0.5*torch.mm(assignments_t, x))
        # x = F.relu(x)
        #
        # # x_comms1 = self.gc1[1](F.dropout(x_comms1, self.dropout, training=self.training))
        # x = self.gc2[1](self.gc1[1](F.dropout(x, self.dropout, training=self.training)), adj)

        # x_comms2 = 0.5*x_comms1 + 0.5*torch.mm(assignments_t, x1)

        x = F.relu(x)
        assignments = self.attn2(x[:-n_communities], x[-n_communities:])# F.softmax(distance, dim=1)
        x = self.gc2(F.dropout(x, self.dropout, training=self.training), adj, assignments)

            #distance = torch.norm(x[:-n_communities].unsqueeze(1) - x[-n_communities:].unsqueeze(0), 2, 2)#F.cosine_similarity(x[:-n_communities].unsqueeze(1), x[-n_communities:].unsqueeze(0), dim=2)#torch.norm(x[:-n_communities].unsqueeze(1) - x[-n_communities:].unsqueeze(0), 2, 2)


            # x_comms1 = self.gc1[1](F.dropout(x_comms1, self.dropout, training=self.training))

            # x = F.relu(x)
            # x = F.dropout(x, self.dropout, training=self.training)
            # x = l1(x)
            # # distance = torch.norm(x[:-n_communities].unsqueeze(1) - x[-n_communities:].unsqueeze(0), 2, 2)
            # # _, assignments = torch.min(distance, dim=1)
            # x = l2(x, adj)#normalize_adj(adj_nonormed, assignments.cpu().data.numpy(), n_communities).cuda())

        # similarity = F.cosine_similarity(exp_posteriors.unsqueeze(1), exp_posteriors.unsqueeze(0), dim=2)
        # similarity[(similarity<0.9).detach()] = 0
        # similarity = similarity / similarity.sum(1, keepdim=True)


        # gmm = GaussianMixture(n_components=n_communities, covariance_type='diag').fit(x.cpu().data.numpy())
        # weights = Variable(torch.FloatTensor(gmm.weights_),requires_grad=False)
        # means = Variable(torch.FloatTensor(gmm.means_),requires_grad=False)
        # precisions = Variable(torch.FloatTensor(gmm.precisions_),requires_grad=False)
        # if self.use_cuda:
        #     weights = weights.cuda()
        #     means = means.cuda()
        #     precisions = precisions.cuda()
        #
        # posteriors = -0.5 * torch.sum((x.unsqueeze(1) - means.unsqueeze(0))**2 * precisions.unsqueeze(0), 2) + torch.log(weights) + 0.5*torch.log(precisions).sum(1)
        # exp_posteriors = F.softmax(posteriors, dim=1)
        # x = x + torch.mm(exp_posteriors, means)*1.0

        return x
