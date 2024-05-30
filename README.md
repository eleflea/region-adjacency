# Region Adjacency

This repository implements region adjacency matrix calculation using NumPy, PyTorch, and CUDA.

A region adjacency matrix indicates which regions are adjacent to each other.

For example, let's consider a 4x4 labeled image (typically from segmentation):

```text
[[0, 0, 1, 1],
 [0, 1, 1, 1],
 [2, 2, 1, 3],
 [2, 2, 3, 4]]
```

This image has 5 classes (ranging from 0 to 4), so the shape of the region adjacency matrix is 5x5.
Using 4-connectivity (immediate neighbors up, down, left, and right), the resulting matrix is:

```text
[[0, 1, 1, 0, 0],
 [1, 0, 1, 1, 0],
 [1, 1, 0, 1, 0],
 [0, 1, 1, 0, 1],
 [0, 0, 0, 1, 0]]
```

Note that all diagonal elements are zero, and adjacent regions are marked as one.

## Usage

There are 4 functions in `region_adjacency.py`.

- `region_adjacency_numpy_loop`: A Simple loop-based NumPy version which is very slow.
- `region_adjacency_numpy`: A vectorized NumPy version.
- `region_adjacency_torch`: A vectorized PyTorch version, similar to the NumPy version.
- `region_adjacency_torch_cpp`: A CUDA version which is very fast.

For detailed usage, please refer to the docstrings of the functions.

## Benchmark

The following table shows the average time used by different functions.

### Experimental Setup

#### Inputs

- Batch size = 8
- Height = Width = 512
- Number of labels = 256
- connectivity = 4-connectivity

#### Software

- PyTorch version: 2.3.0
- NumPy version: 1.26.1
- CUDA version: 12.1

#### Hardware

- CPU: Intel Core i9-13900K
- GPU: NVIDIA GeForce RTX 4090

| Function                      | Device | Avg Time (ms) | Speedup |
| ----------------------------- | ------ | ------------- | ------- |
| `region_adjacency_numpy_loop` | CPU    | 309.36        | 1       |
| `region_adjacency_numpy`      | CPU    | 10.03         | 30.84   |
| `region_adjacency_torch`      | CPU    | 2.05          | 150.9   |
| `region_adjacency_torch`      | GPU    | 7.15          | 43.27   |
| `region_adjacency_torch_cpp`  | GPU    | 0.153         | 2022    |

## License

This project is licensed under the [MIT License](LICENSE).
