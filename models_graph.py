import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv
from dgl.nn import GlobalAttentionPooling
from dgl.nn import MaxPooling

class GNNs(nn.Module):
    def __init__(self, args, n_classes):
        super(GNNs, self).__init__()

        self.args = args
        args.src_vocab_size, args.embed_dim, args.hidden
        if args.model == 'GCN':
            self.conv1 = GraphConv(args.hidden, args.hidden)  
            self.conv2 = GraphConv(args.hidden, args.hidden)  
        elif args.model == 'GAT':
            self.conv1 = GATConv(args.hidden, int(args.hidden/args.GAT_heads),args.GAT_heads)
            self.conv2 = GATConv(int(args.hidden/args.GAT_heads)*args.GAT_heads, int(args.hidden/args.GAT_heads),args.GAT_heads)  
        elif args.model == 'GraphSAGE':
            self.conv1 = SAGEConv(args.hidden, args.hidden,args.GraphSAGE_aggregator) 
            self.conv2 = SAGEConv(args.hidden, args.hidden,args.GraphSAGE_aggregator)  

        hidden_layer = [args.hidden,args.hidden,int(args.hidden/2),int(args.hidden/4),n_classes]
        self.e1 = nn.Linear(args.hidden, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.e5 = nn.Linear(hidden_layer[3], hidden_layer[4])
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.3)
        self.gcn_bn = nn.BatchNorm1d(args.hidden)
        self.batchnorm0 = nn.BatchNorm1d(hidden_layer[0])
        self.batchnorm1 = nn.BatchNorm1d(hidden_layer[1])
        self.batchnorm2 = nn.BatchNorm1d(hidden_layer[2])
        self.batchnorm3 = nn.BatchNorm1d(hidden_layer[3])

        self.src_emb = nn.Embedding(args.src_vocab_size, args.hidden)

        self.gate_nn = nn.Linear(args.hidden, 1)  # the gate layer that maps node feature to scalar
        self.gap = GlobalAttentionPooling(self.gate_nn)
        self.maxpool = MaxPooling()

    def forward(self, g):
        
        with g.local_scope():
            h = self.src_emb(g.ndata['feat'])
            h = F.relu(self.conv1(g, h)) 
            if self.args.model == 'GAT':
                h = h.view(-1, h.size(1) * h.size(2))
            h = F.relu(self.conv2(g, h))
            if self.args.model == 'GAT':
                h = h.view(-1, h.size(1) * h.size(2))
            
            g.ndata['h'] = h    
            hg = dgl.mean_nodes(g,'h') 
            # hg = self.maxpool(g,h) 
            # hg = self.gap(g,h) 
            
        h_1 = F.leaky_relu(self.batchnorm0(self.e1(hg)), negative_slope=0.05, inplace=True)
        h_1 = self.dropout(h_1)
        h_2 = F.leaky_relu(self.batchnorm1(self.e2(h_1)), negative_slope=0.05, inplace=True)
        h_2 = self.dropout(h_2)
        h_3 = F.leaky_relu(self.e3(h_2), negative_slope=0.1, inplace=True)
        h_3 = self.dropout(h_3)
        h_4 = F.leaky_relu(self.e4(h_3), negative_slope=0.1, inplace=True)
        y = self.e5(h_4)
        
        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1)
            # return self.sigmoid(y)
        elif self.args.task_type == 'Regression':
            return y


