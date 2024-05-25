#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void adj_matrix_cuda_forward_kernel(const int64_t *__restrict__ segments,
                                               const scalar_t *__restrict__ features, scalar_t *__restrict__ output_adj,
                                               float sigma, size_t height, size_t width, size_t node_size) {
  constexpr scalar_t one = 1.f;
  const int n_hw = height * width;
  for (int i = threadIdx.x / 4; i < n_hw; i += (blockDim.x / 4)) {
    const int index = blockIdx.x * n_hw + i;
    const int sub_i = threadIdx.x % 4;
    int other_i = index;
    switch (sub_i) {
      case 0:
        if (i >= (int)width) {
          other_i = index - (int)width;
        }
        break;
      case 1:
        if (i + (int)width < n_hw) {
          other_i = index + (int)width;
        }
        break;
      case 2:
        if (i % width != 0) {
          --other_i;
        }
        break;
      case 3:
        if (i % width != (int)width - 1) {
          ++other_i;
        }
        break;

      default:
        break;
    }

    const auto v = segments[index];
    if (const auto nv = segments[other_i]; nv != v) {
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
                               adj_matrix_cuda_forward_kernel<scalar_t>
                                 <<<blocks, threads>>>(segments.data<int64_t>(), features.data<scalar_t>(),
                                                       output_adj.data<scalar_t>(), sigma, height, width, node_size);
                             }));

  const auto f_square = features.pow(2).sum(-1);
  const auto distances = f_square.unsqueeze(-1) + f_square.unsqueeze(1) - 2 * features.bmm(features.transpose(-2, -1));
  output_adj.mul_(distances.div(-sigma * sigma).exp());
  return output_adj;
}
