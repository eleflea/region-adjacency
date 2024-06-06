#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int NEIGHBOR_TABLE[12];

template <typename scalar_t, int neighbor_count>
__global__ void region_adjacency_cuda_forward_kernel(const scalar_t *__restrict__ labelled_imgs,
                                                     uint8_t *__restrict__ output_adj, const size_t label_size,
                                                     const int height, const int width) {
  const auto x = threadIdx.x + blockDim.x * blockIdx.x;
  const auto y = threadIdx.y + blockDim.y * blockIdx.y;
  const auto index = (blockIdx.z * height + y) * width + x;
  if (x < width && y < height) {
    // calculate neighbor index (other_i) based on sub_i
    // -------------
    // | 2 | 1 | 3 |
    // -------------
    // | 0 |   | X |
    // -------------
    // | X | X | X |
    // -------------
    const auto v = labelled_imgs[index];
    for (size_t i = 0; i < neighbor_count; ++i) {
      const auto neighbor = NEIGHBOR_TABLE + i * 3;
      auto other_index = index + (x != neighbor[0] && y != neighbor[1]) * neighbor[2];

      const auto nv = labelled_imgs[other_index];
      const auto pbase = output_adj + blockIdx.z * label_size * label_size;
      const auto pa = pbase + v * label_size + nv;
      const auto pb = pbase + nv * label_size + v;
      if (nv != v && *pa == 0) {
        *pa = 1;
        *pb = 1;
      }
    }
  }
}

torch::Tensor region_adjacency_cuda_forward(const torch::Tensor labelled_imgs, const int num_labels,
                                            const int connectivity) {
  const auto batch_size = labelled_imgs.size(0);
  const int height = labelled_imgs.size(1);
  const int width = labelled_imgs.size(2);
  const size_t label_size = num_labels == 0 ? labelled_imgs.max().item<int64_t>() + 1 : num_labels;
  auto output_adj = torch::zeros({batch_size, num_labels, num_labels}, labelled_imgs.options().dtype(torch::kUInt8));

  constexpr auto tile_size = 32;
  const dim3 threads(tile_size, tile_size);
  const dim3 blocks((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size, batch_size);

  const int neighbor_table[] = {0, -1, -1, -1, 0, -width, 0, 0, -width - 1, width - 1, 0, -width + 1};
  cudaMemcpyToSymbol(NEIGHBOR_TABLE, neighbor_table, 12 * sizeof(int));

  if (connectivity == 1) {
    AT_DISPATCH_INTEGRAL_TYPES(labelled_imgs.type(), "region_adjacency_forward_cuda", ([&] {
                                 region_adjacency_cuda_forward_kernel<scalar_t, 2>
                                   <<<blocks, threads>>>(labelled_imgs.data<scalar_t>(), output_adj.data<uint8_t>(),
                                                         label_size, height, width);
                               }));
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(labelled_imgs.type(), "region_adjacency_forward_cuda", ([&] {
                                 region_adjacency_cuda_forward_kernel<scalar_t, 4>
                                   <<<blocks, threads>>>(labelled_imgs.data<scalar_t>(), output_adj.data<uint8_t>(),
                                                         label_size, height, width);
                               }));
  }

  output_adj = output_adj.to(labelled_imgs.dtype());

  return output_adj;
}
