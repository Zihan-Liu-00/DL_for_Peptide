import dgl
import torch
import networkx as nx
import numpy as np
from dgl.data import DGLDataset
# Convert a sequence peptide data to a graph
# The implementation of peptides in Martini 2 the molecule dynamics software

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

beads = {'Nda':1,'N0':2,'Qa':4,'Qd':3,'SC4':5,'SC5':6,'C3':7,'C5':8,
        'AC1':9,'AC2':10,'P1':11,'P4':12,'P5':13,'SP1':14,'Snd':0,}

# Components of amino acids in beads
amino_acids_feature = {'W':[beads['Nda'],beads['SC4'],beads['Snd'],beads['SC5'],beads['SC5']],
                        'F':[beads['Nda'],beads['SC5'],beads['SC5'],beads['SC5']],
                        'Y':[beads['Nda'],beads['SC4'],beads['SC4'],beads['SP1']],
                        'H':[beads['Nda'],beads['SC4'],beads['SP1'],beads['SP1']],
                        'K':[beads['Nda'],beads['C3'],beads['Qd']],
                        'R':[beads['Nda'],beads['N0'],beads['Qd']],
                        'D':[beads['Nda'],beads['Qa']],
                        'E':[beads['Nda'],beads['Qa']],
                        'V':[beads['Nda'],beads['AC2']],
                        'I':[beads['Nda'],beads['AC1']],
                        'L':[beads['Nda'],beads['AC1']],
                        'M':[beads['Nda'],beads['C5']],
                        'C':[beads['Nda'],beads['C5']],
                        'S':[beads['Nda'],beads['P1']],
                        'T':[beads['Nda'],beads['P1']],
                        'Q':[beads['Nda'],beads['P4']],
                        'N':[beads['Nda'],beads['P5']],
                        'P':[beads['N0'],beads['C3']],
                        'A':[beads['N0']],'G':[beads['Nda']]}

# Connection between beads in amino acids
amino_acids_connection = {'W':[[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,1],[0,1,1,0,1],[0,0,1,1,0]],
                        'F':[[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]],
                        'Y':[[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]],
                        'H':[[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]],
                        'K':[[0,1,0],[1,0,1],[0,1,0]],'R':[[0,1,0],[1,0,1],[0,1,0]],
                        'D':[[0,1],[1,0]],'E':[[0,1],[1,0]],'V':[[0,1],[1,0]],
                        'I':[[0,1],[1,0]],'L':[[0,1],[1,0]],'M':[[0,1],[1,0]],
                        'C':[[0,1],[1,0]],'S':[[0,1],[1,0]],'T':[[0,1],[1,0]],
                        'Q':[[0,1],[1,0]],'N':[[0,1],[1,0]],'P':[[0,1],[1,0]],
                        'A':[[0]],'G':[[0]]}

def make_data(features):
    graph_attrs = []
    graph_adjas = []
    dataset = []
    for i in range(len(features)):
        graph_attr = []
        count = 0
        for n in list(features[i]):
            if count == 0:
                graph_attr = graph_attr + ([beads['Qd']] + amino_acids_feature[n][1:])
            elif count == len(features[i])-1:
                graph_attr = graph_attr + ([beads['Qa']] + amino_acids_feature[n][1:])
            else:
                graph_attr = graph_attr + amino_acids_feature[n]
            count = count + 1
        
        graph_adja = np.zeros((len(graph_attr),len(graph_attr)))
        position = 0
        last_backbone = 0
        for n in list(features[i]):
            graph_adja[position:position+len(amino_acids_connection[n]),position:position+len(amino_acids_connection[n])] = np.array(amino_acids_connection[n])
            # Add connections between backbones
            if position:
                graph_adja[position,last_backbone]=1
                graph_adja[last_backbone,position]=1
            last_backbone = position
            position = position+len(amino_acids_connection[n])
        for n in range(len(graph_attr)):
            graph_adja[n,n]=1
        g = nx.Graph(graph_adja)
        for node in range(g.number_of_nodes()):
            g.nodes[node]['feat'] = torch.tensor(graph_attr[node], dtype=torch.long)
        g = dgl.from_networkx(g, node_attrs=['feat'])
        dataset.append(g)
    return dataset

class MyDataSet(DGLDataset):
    def __init__(self,graphs,labels,url=None,
                raw_dir=None,
                save_dir=None,
                force_reload=False,
                verbose=False):
        super(MyDataSet, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        self.graphs = graphs
        self.labels = labels
    def process(self):
        return
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
        
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    if labels[0].dtype == torch.int64:
        return dgl.batch(graphs).to(device), torch.tensor(labels).to(device)
    else:
        return dgl.batch(graphs).to(device), torch.tensor(labels).unsqueeze(1).to(device)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    # sample_num = labels.shape[0]
    # pred_labels = np.where(output.cpu()<0.5,0,1)
    # acc = (pred_labels==labels.cpu().numpy()).sum()/sample_num
    # return acc

