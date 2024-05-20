import torch
from torch.utils.cpp_extension import load

adj_matrix_cpp = load(name='adj_matrix', sources=['adj_matrix_cuda.cpp', 'adj_matrix_cuda_kernel.cu'])
adj_matrix = adj_matrix_cpp.forward
seg = torch.as_tensor([
    [0, 0, 0, 1],
    [0, 1, 1, 1],
    [2, 2, 3, 3],
    [2, 2, 3, 3]
], dtype=torch.int64).cuda()
features = torch.randn(1, 6, 5, dtype=torch.float32).cuda()
adj_torch = adj_matrix(seg.view(1, 4, 4), features, 5).cpu().numpy()
print(adj_torch)
