import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.init as init

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

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution_P1(Module):
    def __init__(self, in_features, out_features, cuda, bias=False):
        super(GraphConvolution_P1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if cuda:
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.weight.data)

    def forward(self, input):
        support = torch.mm(input, self.weight)
        return support

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution_P2(Module):
    def __init__(self):
        super(GraphConvolution_P2, self).__init__()

    def forward(self, support, adj):
        output = SparseMM(adj)(support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
