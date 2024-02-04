
We provide a peptide self-assembly dataset in this project. On this dataset, we provide graph and sequence encoding and modeling methods for peptides as a tool library. To implement this code, you mainly need to install **Python**, **PyTorch** (https://pytorch.org/get-started/locally/), and **DGL** (https://docs.dgl.ai/en/latest/install/) on your device. 

***Guide->***

To download the peptide dataset in form of graphs, please visit:  
https://drive.google.com/file/d/1CQnVhISNXCl3siHDB1KZEAEhVpopVcTV/view?usp=sharing  
The .zip file should be extracted and placed in the root directory.

``main_mlp.py`` is the running file of encoding peptides' sequence information into **one-dimensional vectors**. Each amino acid is mapped to a word embedding and spliced to form a vector representing the peptide. This form of data can be used to predict labels using MLP, SVM, and other models.

``main_seq.py`` is the running file of encoding peptides' sequence information into **sequential representations**. The peptides are encoded into matrices in the shape of sequence length × embedding dimension, i.e., the embeddings of amino acids are arranged by their positions. This form of data can be used to predict labels using RNN, LSTM, Transformer, *etc*.

``main_graph.py`` is the running file of encoding peptides' chemical structure into **homogeneous undirected graphs**. We model the peptide molecules according to the coarse-graining method from Martini 3, in which nodes represent beads and edges represent chemical bonds. Geometric Models, such as GCN, GAT, and GraphSAGE, can be used to extract the graph representations from the peptides' chemical structures and, thus, for predictions.

*Note*: The current code only supports the modeling of sequence peptides composed of natural amino acids. If you want to use this code for other peptides (modifications, cyclic, *etc*.) but don't know how to get started, you can send an email or leave an issue.

***Reference->***

For more information on the principles and details of modeling, as well as a comparison of the performance of each modeling method, please visit https://academic.oup.com/bib/article-abstract/24/6/bbad409/7424447?redirectedFrom=fulltext and refer to our publication:

Liu, Zihan, Jiaqi Wang, Yun Luo, Shuang Zhao, Wenbin Li, and Stan Z. Li. "**Efficient prediction of peptide self-assembly through sequential and graphical encoding.**" Briefings in Bioinformatics 24, no. 6 (2023): bbad409.

For an exploration of the peptide self-assembly mechanism based on this model, please visit https://onlinelibrary.wiley.com/doi/full/10.1002/advs.202301544 and see our publication:

Wang, Jiaqi, Zihan Liu, Shuang Zhao, Tengyan Xu, Huaimin Wang, Stan Z. Li, and Wenbin Li. "**Deep Learning Empowers the Discovery of Self‐Assembling Peptides with Over 10 Trillion Sequences.**" Advanced Science 10, no. 31 (2023): 2301544.

If you are interested in the AP of a mixed pentapeptide system, please refer to https://github.com/wangwestlake/PDMP-PEP-SELFASSEMBLY.

***Information->***

Please contact zihanliu@hotmail.com or liuzihan@westlake.edu.cn if you are interested in this code project or if you have issues to discuss. If you are interested in the molecular dynamics part (i.e., data collection) and the mechanism of self-assembly, please contact wangjiaqi@westlake.edu.cn.
