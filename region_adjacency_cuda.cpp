#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor region_adjacency_cuda_forward(const torch::Tensor segments, const int num_labels,
                                      const int connectivity);

torch::Tensor region_adjacency_forward(const torch::Tensor segments, const int num_labels = 0,
                                 const int connectivity = 1) {
  CHECK_INPUT(segments);

  return region_adjacency_cuda_forward(segments, num_labels, connectivity);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &region_adjacency_forward, "region adjacency matrix forward (CUDA)"); }
