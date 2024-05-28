#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void region_adjacency_cuda_forward_kernel(const int64_t *__restrict__ labelled_imgs,
                                                     scalar_t *__restrict__ output_adj, const size_t label_size,
                                                     const int neighbor_count, const int height, const int width) {
  constexpr scalar_t one = 1;
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

    const auto v = labelled_imgs[index];
    if (const auto nv = labelled_imgs[other_i]; nv != v) {
      const auto t = blockIdx.x * label_size;
      output_adj[(t + v) * label_size + nv] = one;
      output_adj[(t + nv) * label_size + v] = one;
    }
  }
}

torch::Tensor region_adjacency_cuda_forward(const torch::Tensor labelled_imgs, const int num_labels,
                                            const int connectivity) {
  const auto batch_size = labelled_imgs.size(0);
  const int height = labelled_imgs.size(1);
  const int width = labelled_imgs.size(2);
  const size_t label_size = num_labels == 0 ? labelled_imgs.max().item<int64_t>() + 1 : num_labels;
  const auto neighbor_count = connectivity == 1 ? 4 : 8;
  auto output_adj = torch::zeros({batch_size, num_labels, num_labels}, labelled_imgs.options());

  const int threads = 1024;
  const int blocks = batch_size;

  AT_DISPATCH_INTEGRAL_TYPES(labelled_imgs.type(), "region_adjacency_forward_cuda", ([&] {
                               region_adjacency_cuda_forward_kernel<scalar_t>
                                 <<<blocks, threads>>>(labelled_imgs.data<int64_t>(), output_adj.data<scalar_t>(),
                                                       label_size, neighbor_count, height, width);
                             }));

  return output_adj;
}
