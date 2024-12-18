import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def multiomics_data():
    # protein_fea
    DC = np.genfromtxt("data-human/02protein_DC.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    protein = DC[:, 0]
    protein_number = len(protein)
    DC = np.array(DC)
    DC = scale(np.array(DC[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    DC = pca.fit_transform(DC)
    DC_adj = sim_graph(DC, protein_number)

    ESM = np.genfromtxt("data-human/02protein_ESM.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    protein = ESM[:, 0]
    protein_number = len(protein)
    ESM = np.array(ESM)
    ESM = scale(np.array(ESM[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    ESM = pca.fit_transform(ESM)
    ESM_adj = sim_graph(ESM, protein_number)

    fusion_protein_fea = np.concatenate((DC, ESM), axis=1)
    fusion_protein_adj = np.logical_or(DC_adj, ESM_adj).astype(int)

    # drug_fea
    PC = np.genfromtxt("data-human/02drug_phy_chem.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    drug = PC[:, 0]
    drug_number = len(drug)
    PC = np.array(PC)
    PC = scale(np.array(PC[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    PC = pca.fit_transform(PC)
    PC_adj = sim_graph(PC, drug_number)

    MACCS = np.genfromtxt("data-human/02drug_MACCS.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    drug = MACCS[:, 0]
    drug_number = len(drug)
    MACCS = np.array(MACCS)
    MACCS = scale(np.array(MACCS[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    MACCS = pca.fit_transform(MACCS)
    MACCS_adj = sim_graph(MACCS, drug_number)

    fusion_drug_fea = np.concatenate((PC, MACCS), axis=1)
    fusion_drug_adj = np.logical_or(PC_adj, MACCS_adj).astype(int)

    #加载label
    labellist = []
    # with open('data-human/03label666.txt', 'r') as file:
    with open('data-human/03label_unique.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # 移除行首和行尾的空白字符
        elements = line.split(" ")  # 使用空格分隔元素
        processed_elements = [int(elements[1]), int(elements[0]), int(elements[2])]  # 互换位置
        labellist.append(processed_elements)
    labellist = torch.Tensor(labellist)
    print("drug protein lable:", labellist.shape)

    protein_feat, protein_adj = torch.FloatTensor(fusion_protein_fea), torch.FloatTensor(fusion_protein_adj)
    drug_feat, drug_adj = torch.FloatTensor(fusion_drug_fea), torch.FloatTensor(fusion_drug_adj)
    return protein_feat, protein_adj, drug_feat, drug_adj, labellist

def sim_graph(omics_data, protein_number):
    sim_matrix = np.zeros((protein_number, protein_number), dtype=float)
    adj_matrix = np.zeros((protein_number, protein_number), dtype=float)

    for i in range(protein_number):
        for j in range(i + 1):
            sim_matrix[i, j] = np.dot(omics_data[i], omics_data[j]) / (
                        np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j, i] = sim_matrix[i, j]

    for i in range(protein_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i, j] = 1
    return adj_matrix