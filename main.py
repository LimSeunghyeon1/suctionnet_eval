import torch
from torch import nn, Tensor
from point_transformer_pytorch import PointTransformerLayer
from torch_cluster import fps, knn_graph






# attn = PointTransformerLayer(
#     dim = 128,
#     pos_mlp_hidden_dim = 64,
#     attn_mlp_hidden_mult = 4,
#     num_neighbors = 16          # only the 16 nearest neighbors would be attended to for each point
# )
# #
# x = torch.randn(1, 2048, 128)
# pos = torch.randn(1, 2048, 3)
# mask = torch.ones(1, 2048).bool()
# sampled_x = []
# for batch in range(x.shape[0]):
#     idx = fps(x[batch], ratio=0.5, random_start=False)
#     idxary = [False for i in range(x[batch].shape[0])]
#     for i in idx:
#         idxary[i] = True
#     new_tens = x[batch][idxary]
#     sampled_x.append(new_tens)
# sampled_x = torch.stack(sampled_x,0)
# print(sampled_x.shape)
# print (attn(feats, pos, mask = mask)) # (1, 16, 128)


x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
#batch = torch.tensor([0, 0, 0, 0])
edge_index = knn_graph(x, k=3, loop=False)

print(edge_index)

