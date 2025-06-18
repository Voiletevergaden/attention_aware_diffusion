import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from src.utils import evaluate, extract_Edges
from src.Model import VAE_EAD
from src.picture import show_picture

history = {
        'total_loss': [],
        'mse_loss': [],
        'kl_loss': [],
        'sparse_loss': []
    }

class celltype_GRN_model():
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def initalize_A(self, data):
        num_genes = data.shape[1]  # 获取基因数量
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (
                np.random.rand(num_genes * num_genes) * 0.0002).reshape(  # 初始化矩阵
            [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0  # 将对角线上的值设置为0，表示不对自己产生调控
        return A

    def init_data(self, ):  # 初始化数据的函数，数据标准化、生成训练数据、构造掩码矩阵
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)  # 真实的基因调控，数据格式可能需要再研究，可能是（gene1,gene2）?
        data = sc.read(self.opt.data_file)  # scanpy包可读取.h5ad的文件
        gene_name = list(data.var_names)  # 获取基因名和表达式
        data_values = data.X  # 数据的值（矩阵）
        Dropout_Mask = (data_values != 0).astype(float)  # 构造掩码？
        means = []
        stds = []
        for i in range(data_values.shape[1]):  # 对矩阵的每列，基于非零值计算mean和std
            tmp = data_values[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())  # 可以直接进行处理
        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 0
        stds[np.isinf(stds)] = 0  # 对异常值进行处理
        data_values = (data_values - means) / stds
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values, -10)
        data_values = np.minimum(data_values, 10)  # 标准化、
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)  # 转换为dataframe，行为细胞、列为基因

        TF = set(Ground_Truth['Gene1'])  # 掩码矩阵，每一行表示一个基因调控对，TF(transcription factors)转录因子
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])  # 所有基因合并为一个全集，表示再调控网络中出现的所有基因

        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])  # 初始化两个掩码矩阵
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):  # 可迭代对象，对于基因i
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:  # j是调控基因，i是被调控，说明这个基因有调控能力，但也曾经被调控
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:  # j具有调控其他基因的能力
                    TF_mask[i, j] = 1
        # TF_mask[i, j] = 1 ⟹ j 是 TF，i 任意（除了自己）;Evaluate_Mask[i, j] = 1 ⟹ j 是 TF，i 在网络里出现过（实际调控过的可能目标）
        feat_train = torch.FloatTensor(data.values)  # 归一化处理后的基因表达矩阵
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))  # 创建一个训练数据集对象，索引(0,n-1)，dropout掩码(哪些位置原始表达值是非零)
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        # 构造基因调控网络的邻接矩阵
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1  # 遍历每一对调控关系，被调控是行，调控者是列
        A_truth = truth_df.values  # (n,n)
        idx_rec, idx_send = np.where(A_truth)  # 返回所有为1的坐标，
        truth_edges = set(zip(idx_send, idx_rec))  # send是调控者，rec是被调控者
        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name,

    def train_model(self):
        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name = self.init_data()
        adj_A_init = self.initalize_A(data)  # 初始的基因调控网络的邻接矩阵
        vae = VAE_EAD(adj_A_init, 1, self.opt.n_hidden, self.opt.K).float().cuda()  # 自定义的变分自编码器模型

        Tensor = torch.cuda.FloatTensor
        optimizer = optim.RMSprop(vae.parameters(), lr=self.opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=self.opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.gamma)
        best_Epr = 0
        vae.train()

        print(vae)
        for epoch in range(self.opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            if epoch % (self.opt.K1 + self.opt.K2) < self.opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                optimizer.zero_grad()
                inputs, data_id, dropout_mask = data_batch
                inputs = Variable(inputs.type(Tensor))
                data_ids.append(data_id.cpu().detach().numpy())
                temperature = max(0.95 ** epoch, 0.5)
                loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs, dropout_mask=dropout_mask.cuda(),
                                                                           temperature=temperature, opt=self.opt)
                sparse_loss = self.opt.alpha * torch.mean(torch.abs(vae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_gauss.item() + loss_cat.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (self.opt.K1 + self.opt.K2) < self.opt.K1:
                    optimizer.step()
                else:
                    optimizer2.step()
            scheduler.step()
            history['total_loss'].append(np.mean(loss_all))
            history['mse_loss'].append(np.mean(mse_rec))
            history['kl_loss'].append(np.mean(loss_kl))
            history['sparse_loss'].append(np.mean(loss_sparse))
            if epoch % (self.opt.K1 + self.opt.K2) >= self.opt.K1:
                Ep, Epr = evaluate(vae.adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                best_Epr = max(Epr, best_Epr)
                print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                      np.mean(loss_sparse))
        df = pd.DataFrame(history)
        df.to_csv('history.csv', index=False)
        show_picture(history)
        extract_Edges(vae.adj_A.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
            self.opt.save_name + '/GRN_inference_result.tsv', sep='\t', index=False)

