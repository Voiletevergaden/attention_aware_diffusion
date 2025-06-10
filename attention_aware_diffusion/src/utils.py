import numpy as np
import pandas as pd
import copy


def extract_Edges(m, geneNames, TFmask):
    geneNames = np.array(geneNames)  # genename是tf和target的汇总
    mat = copy.deepcopy(m)  # mat的结构可能是（target，tf）这样的矩阵，
    num_nodes = mat.shape[0]  # mat矩阵的行数n，得到n*n的（0，1）矩阵
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        mat = mat * TFmask  # 结果为原来或者是0
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)  # 查找在矩阵中有价值的（为1的）横纵坐标，横坐标为target，纵坐标为tf
    edges_df = pd.DataFrame(
        {'TF': geneNames[idx_send], 'Target': geneNames[idx_rec], 'WeightOfEdge': (mat[idx_rec, idx_send])}
    )
    edges_df = edges_df.sort_values('WeightOfEdge', ascending=False)

    return edges_df


def evaluate(A, truth_edges, Evaluate_Mask):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A = abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    A_val = list(np.sort(abs(A.reshape(-1,1)),0)[:,0])
    A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    A_indicator_all = np.zeros([num_nodes,num_nodes])
    A_indicator_all[abs(A)>cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
    overlap_A = A_edges.intersection(truth_edges)
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))