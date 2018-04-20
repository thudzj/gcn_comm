import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.init as init
from torch.autograd.variable import Variable
import torch.nn.functional as F

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, cuda, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if cuda:
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            if cuda:
                self.bias = Parameter(torch.cuda.FloatTensor(out_features))
            else:
                self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.weight.data)
        if self.bias is not None:
            init.constant(self.bias.data, 0)

    def forward(self, input, adj, assignments):
        n_communities = assignments.shape[1]
        support = torch.mm(input, self.weight)
        output1 = torch.mm(assignments/assignments.sum(1, keepdim=True), support[-n_communities:]) + SparseMM(adj)(support[:-n_communities])
        output2 = torch.mm(assignments.transpose(0, 1)/assignments.transpose(0, 1).sum(1, keepdim=True), support[:-n_communities])
        #output2 /= output2.sum(1, keepdim=True)
        return torch.cat((output1, output2))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Attention(Module):
    def __init__(self, in_features, hidden_atten, cuda, bias=False):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.hidden_atten = hidden_atten
        if cuda:
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, hidden_atten))
            self.alpha = Parameter(torch.cuda.FloatTensor(2 * hidden_atten, 1))
        else:
            self.weight = Parameter(torch.Tensor(in_features, hidden_atten))
            self.alpha = Parameter(torch.Tensor(2 * hidden_atten, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.weight.data)
        init.xavier_uniform(self.alpha.data)

    def forward(self, input1, input2):
        sp1 = input1.shape[0]
        sp2 = input2.shape[0]
        f1 = torch.nn.LeakyReLU(0.2)(torch.mm(input1, self.weight))
        #f1 = f1.unsqueeze(1).expand([sp1, sp2, self.hidden_atten])
        f2 = torch.nn.LeakyReLU(0.2)(torch.mm(input2, self.weight))
        #f2 = f2.unsqueeze(0).expand([sp1, sp2, self.hidden_atten])
        #support = torch.matmul(torch.cat((f1, f2), 2), self.alpha).squeeze()
        support = F.sigmoid(-torch.norm(f1.unsqueeze(1) - f2.unsqueeze(0), 2, 2))#F.cosine_similarity(f1.unsqueeze(1), f2.unsqueeze(0), dim=2)
        return support

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
