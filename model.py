import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#GAT
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903. This part of code refers to the implementation of https://github.com/Diego999/pyGAT.git

    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #print("Here GAL")
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # （N，N）

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return F.relu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class selfattention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return output

#GCN
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        output = torch.matmul(adj, Wh)
        return F.relu(output)

#GAC
class GraphAttentionLayer_GAC(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer_GAC, self).__init__()
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*h.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        attention = F.softmax(e, dim=1)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class GAC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAC, self).__init__()
        self.gc1 = GraphAttentionLayer_GAC(input_dim, hidden_dim)
        self.gc2 = GraphAttentionLayer_GAC(hidden_dim, output_dim)

    def forward(self, input, adj):
        x = F.relu(self.gc1(input, adj))
        x = self.gc2(x, adj)
        return x

class MAGNETDTI(nn.Module):

    def __init__(self, nprotein, ndrug, nproteinfeat, ndrugfeat, nhid, nheads, alpha):
        """Dense version of GAT."""
        super(MAGNETDTI, self).__init__()
        #print("Here MAGNETDTI")
        self.protein_attentions1 = [GraphAttentionLayer(nproteinfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_attentions1):
            self.add_module("Attention_Protein1_{}".format(i), attention)
        self.protein_MultiHead1 = [selfattention(64,nheads) for _ in range(nheads)]
        #self.protein_MultiHead1 = [selfattention(nprotein, nhid, nprotein) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_MultiHead1):
            self.add_module("Self_Attention_Protein1_{}".format(i), attention)

        self.protein_prolayer1 = nn.Linear((nhid * nheads), (nhid * nheads), bias=False)
        self.protein_LNlayer1 = nn.LayerNorm(nhid * nheads)
        self.protein_attentions2 = [GraphAttentionLayer((nhid * nheads), nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_attentions2):
            self.add_module("Attention_Protein2_{}".format(i), attention)

        self.protein_MultiHead2 = [selfattention(64,nheads) for _ in range(nheads)]
        #self.protein_MultiHead2 = [selfattention(nprotein, nhid, nprotein) for _ in range(nheads)]
        for i, attention in enumerate(self.protein_MultiHead2):
            self.add_module("Self_Attention_Protein2_{}".format(i), attention)

        self.protein_prolayer2 = nn.Linear((nhid * nheads), (nhid * nheads), bias=False)
        self.protein_LNlayer2 = nn.LayerNorm(nhid * nheads)
        self.drug_attentions1 = [GraphAttentionLayer(ndrugfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions1):
            self.add_module("Attention_Drug1_{}".format(i), attention)

        self.drug_MultiHead1 = [selfattention(64,nheads) for _ in range(nheads)]
        #self.drug_MultiHead1 = [selfattention(ndrug, nhid, ndrug) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_MultiHead1):
            self.add_module("Self_Attention_Drug1_{}".format(i), attention)

        self.drug_prolayer1 = nn.Linear((nhid * nheads), (nhid * nheads), bias=False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid * nheads)
        self.drug_attentions2 = [GraphAttentionLayer((nhid * nheads), nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions2):
            self.add_module("Attention_Drug2_{}".format(i), attention)

        self.drug_MultiHead2 = [selfattention(64,nheads) for _ in range(nheads)]
        #self.drug_MultiHead2 = [selfattention(ndrug, nhid, ndrug) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_MultiHead2):
            self.add_module("Self_Attention_Drug2_{}".format(i), attention)

        self.drug_prolayer2 = nn.Linear((nhid * nheads), (nhid * nheads), bias=False)
        self.drug_LNlayer2 = nn.LayerNorm(nhid * nheads)
        self.FClayer1 = nn.Linear(nhid * nheads * 2, nhid * nheads * 2)
        self.FClayer2 = nn.Linear(nhid * nheads * 2, nhid * nheads * 2)
        self.FClayer3 = nn.Linear(nhid * nheads * 2, 1)
        self.output = nn.Sigmoid()

    def forward(self, protein_features, protein_adj, drug_features, drug_adj, idx_protein_drug, device):
        proteinx = torch.cat([att(protein_features, protein_adj) for att in self.protein_attentions1], dim=1)
        proteinx = self.protein_prolayer1(proteinx)
        #print("Here")
        proteinayer = proteinx
        temp = torch.zeros_like(proteinx)
        for selfatt in self.protein_MultiHead1:
            #print(f"Here1, proteinx.shape: {proteinx.shape}")
            temp = temp + selfatt(proteinx.unsqueeze(0))
        #print(f"{proteinx.shape}")
        proteinx = temp + proteinayer
        #print("Here2")
        proteinx = self.protein_LNlayer1(proteinx)
        #print(f"{proteinx.shape}, {protein_adj.shape}")
        proteinx = torch.cat([att(proteinx[0], protein_adj) for att in self.protein_attentions2], dim=1)
        proteinx = self.protein_prolayer2(proteinx)
        proteinayer = proteinx
        temp = torch.zeros_like(proteinx)
        for selfatt in self.protein_MultiHead2:
            temp = temp + selfatt(proteinx.unsqueeze(0))
        proteinx = temp + proteinayer
        #print("Here3")
        proteinx = self.protein_LNlayer2(proteinx)
        drugx = torch.cat([att(drug_features, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = self.drug_prolayer1(drugx)
        druglayer = drugx
        temp = torch.zeros_like(drugx)
        for selfatt in self.drug_MultiHead1:
            temp = temp + selfatt(drugx.unsqueeze(0))

        drugx = temp + druglayer
        drugx = self.drug_LNlayer1(drugx)
        drugx = torch.cat([att(drugx[0], drug_adj) for att in self.drug_attentions2], dim=1)
        drugx = self.drug_prolayer2(drugx.unsqueeze(0))
        druglayer = drugx
        temp = torch.zeros_like(drugx)
        #print(f"Here4: drugx.shape {drugx.shape}")
        for selfatt in self.drug_MultiHead2:
            temp = temp + selfatt(drugx)
        drugx = temp + druglayer
        drugx = self.drug_LNlayer2(drugx)
        #print("drug:", idx_protein_drug[:, 1])
        #print("protein:", idx_protein_drug[:, 0])
        #print("Here5")
        #print(f"proteinx.shape: {proteinx.shape}, drugx.shape: {drugx.shape}")
        proteinx = proteinx.squeeze(0)
        drugx = drugx.squeeze(0)
        protein_drug_x = torch.cat((proteinx[idx_protein_drug[:, 0]], drugx[idx_protein_drug[:, 1]]), dim=1)#------------>Error Here
        protein_drug_x = protein_drug_x.to(device)
        protein_drug_x = self.FClayer1(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer2(protein_drug_x)
        protein_drug_x = F.relu(protein_drug_x)
        protein_drug_x = self.FClayer3(protein_drug_x)
        protein_drug_x = protein_drug_x.squeeze(-1)
        return protein_drug_x