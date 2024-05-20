#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor adj_matrix_cuda_forward(torch::Tensor segments, torch::Tensor features, float sigma);

torch::Tensor adj_matrix_forward(torch::Tensor segments, torch::Tensor features, float sigma) {
  CHECK_INPUT(segments);
  CHECK_INPUT(features);

  return adj_matrix_cuda_forward(segments, features, sigma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &adj_matrix_forward, "adj matrix forward (CUDA)"); }
