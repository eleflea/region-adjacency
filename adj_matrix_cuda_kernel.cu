#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void adj_matrix_cuda_forward_kernel(const int64_t *__restrict__ segments,
                                               const scalar_t *__restrict__ features,
                                               scalar_t *__restrict__ output_adj, float sigma, size_t height,
                                               size_t width, size_t node_size) {
  constexpr scalar_t one = 1.f;
  const int n_hw = height * width;
  for (int i = threadIdx.x; i < n_hw; i += blockDim.x) {
    const int index = blockIdx.x * n_hw + i;
    auto up = index - (int)width;
    up = up < 0 ? index : up;
    auto down = index + (int)width;
    down = down >= n_hw ? index : down;
    auto left = index % width == 0 ? index : index - 1;
    auto right = index + 1;
    right = right % width == 0 ? index : right;

    const auto v = segments[index];
    if (const auto nv = segments[up]; nv != v) {
      output_adj[blockIdx.x * node_size * node_size + v * node_size + nv] = one;
      output_adj[blockIdx.x * node_size * node_size + nv * node_size + v] = one;
    }
    if (const auto nv = segments[down]; nv != v) {
      output_adj[blockIdx.x * node_size * node_size + v * node_size + nv] = one;
      output_adj[blockIdx.x * node_size * node_size + nv * node_size + v] = one;
    }
    if (const auto nv = segments[left]; nv != v) {
      output_adj[blockIdx.x * node_size * node_size + v * node_size + nv] = one;
      output_adj[blockIdx.x * node_size * node_size + nv * node_size + v] = one;
    }
    if (const auto nv = segments[right]; nv != v) {
      output_adj[blockIdx.x * node_size * node_size + v * node_size + nv] = one;
      output_adj[blockIdx.x * node_size * node_size + nv * node_size + v] = one;
    }
  }
}

torch::Tensor adj_matrix_cuda_forward(torch::Tensor segments, torch::Tensor features, float sigma) {
  const auto batch_size = segments.size(0);
  const auto height = segments.size(1);
  const auto width = segments.size(2);
  const auto node_size = features.size(1);
  auto output_adj = torch::zeros({batch_size, node_size, node_size}, features.options());

  const int threads = 1024;
  const int blocks = batch_size;

  AT_DISPATCH_FLOATING_TYPES(features.type(), "adj_matrix_forward_cuda", ([&] {
                               adj_matrix_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                 segments.data<int64_t>(), features.data<scalar_t>(), output_adj.data<scalar_t>(),
                                 sigma, height, width, node_size);
                             }));

  const auto f_square = features.pow(2).sum(-1);
  const auto distances = f_square.unsqueeze(-1) + f_square.unsqueeze(1) - 2 * features.bmm(features.transpose(-2, -1));
  output_adj.mul_(distances.div(-sigma * sigma).exp());
  return output_adj;
}
