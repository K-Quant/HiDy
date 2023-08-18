import  torch
from    torch import nn
from    torch.nn import functional as F
from    utils import sparse_dropout, dot


class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless


        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        x, support = inputs
        if self.training and self.is_sparse_inputs:
            print("not support")
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x.to(torch.float64), self.weight.to(torch.float64))
        else:
            xw = self.weight

        out = torch.sparse.mm(support.to(torch.float64), xw.to(torch.float64))

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

