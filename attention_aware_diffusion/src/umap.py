import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 vae.adj_A 已经是训练结束的邻接矩阵
#adj_matrix = vae.adj_A.cpu().detach().numpy()

# 保存邻接矩阵，基因名对齐
#adj_df = pd.DataFrame(adj_matrix, index=gene_name, columns=gene_name)
#adj_df.to_csv(self.opt.save_name + '/adjacency_matrix.txt', sep='\t')

#print(f"邻接矩阵已保存至 {self.opt.save_name + '/adjacency_matrix.txt'}")

# ------------------ 生成 Degree 特征 ------------------
#degree_features = np.sum(adj_matrix, axis=1).reshape(-1, 1)

# ------------------ UMAP 降维 ------------------
reducer = umap.UMAP(random_state=42)
#embedding = reducer.fit_transform(degree_features)

# ------------------ 可视化 ------------------
plt.figure(figsize=(8, 6))
#sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], s=50)
plt.title("UMAP of Gene Degree Features")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.show()
