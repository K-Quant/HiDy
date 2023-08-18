import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution

from    config import args

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim
        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden,
                                                     activation=F.elu, #activation=F.relu
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    GraphConvolution(args.hidden, args.hidden,
                                                     activation=F.elu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    )
        self.predict = nn.Linear(args.hidden, self.output_dim)

    def forward(self, inputs):
        x, support = inputs
        x = self.layers((x, support))
        x = self.predict(x[0])
        return torch.sigmoid(x)


    def l2_loss(self):
        layer = self.layers.children()
        layer = next(iter(layer))
        loss = None
        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
