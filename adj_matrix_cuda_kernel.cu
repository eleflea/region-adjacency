#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void adj_matrix_cuda_forward_kernel(const int64_t *__restrict__ segments,
                                               const scalar_t *__restrict__ features, scalar_t *__restrict__ output_adj,
                                               const float sigma, const int neighbor_count, const int height,
                                               const int width, const size_t node_size) {
  constexpr scalar_t one = 1.f;
  const int n_hw = height * width;
  for (int i = threadIdx.x / neighbor_count; i < n_hw; i += (blockDim.x / neighbor_count)) {
    const int index = blockIdx.x * n_hw + i;

    // calculate neighbor index (other_i) based on sub_i
    // -------------
    // | 4 | 0 | 5 |
    // -------------
    // | 3 |   | 1 |
    // -------------
    // | 7 | 2 | 6 |
    // -------------
    const int sub_i = threadIdx.x % neighbor_count;
    int other_i = index;
    switch (sub_i) {
      case 0:
        if (i >= width) {
          other_i -= width;
        }
        break;
      case 1:
        if (i % width != width - 1) {
          ++other_i;
        }
        break;
      case 2:
        if (i + width < n_hw) {
          other_i += width;
        }
        break;
      case 3:
        if (i % width != 0) {
          --other_i;
        }
        break;
      case 4:
        if (i >= width && i % width != 0) {
          other_i -= width + 1;
        }
        break;
      case 5:
        if (i >= width && i % width != width - 1) {
          other_i -= width - 1;
        }
        break;
      case 6:
        if (i + width < n_hw && i % width != width - 1) {
          other_i += width + 1;
        }
        break;
      case 7:
        if (i + width < n_hw && i % width != 0) {
          other_i += width - 1;
        }
        break;

      default:
        break;
    }

    const auto v = segments[index];
    if (const auto nv = segments[other_i]; nv != v) {
      const auto t = blockIdx.x * node_size;
      output_adj[(t + v) * node_size + nv] = one;
      output_adj[(t + nv) * node_size + v] = one;
    }
  }
}

torch::Tensor adj_matrix_cuda_forward(const torch::Tensor segments, const torch::Tensor features, const float sigma,
                                      const int connectivity) {
  const auto batch_size = segments.size(0);
  const int height = segments.size(1);
  const int width = segments.size(2);
  const auto node_size = features.size(1);
  const auto neighbor_count = connectivity == 1 ? 4 : 8;
  auto output_adj = torch::zeros({batch_size, node_size, node_size}, features.options());

  const int threads = 1024;
  const int blocks = batch_size;

  AT_DISPATCH_FLOATING_TYPES(features.type(), "adj_matrix_forward_cuda", ([&] {
                               adj_matrix_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                 segments.data<int64_t>(), features.data<scalar_t>(), output_adj.data<scalar_t>(),
                                 sigma, neighbor_count, height, width, node_size);
                             }));

  const auto f_square = features.pow(2).sum(-1);
  const auto distances = f_square.unsqueeze(-1) + f_square.unsqueeze(1) - 2 * features.bmm(features.transpose(-2, -1));
  output_adj.mul_(distances.div(-sigma * sigma).exp());
  return output_adj;
}
