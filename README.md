# Minuet

Minuet is a library that efficiently implements sparse 
convolutions for point clouds on GPUs.

#### [Documentation](https://uoft-ecosystem.github.io/Minuet) | [Research Paper](https://www.cs.toronto.edu/~jcyang/minuet.pdf) | [Artifact Evaluation](https://github.com/Kipsora/MinuetArtifacts)

## Introduction

Sparse Convolution (SC) is widely used for processing 3D point clouds that are 
inherently sparse. Different from dense convolution, SC preserves the sparsity 
of the input point cloud by only allowing outputs to specific locations.
To efficiently compute SC, prior SC engines first use hash tables to build a 
kernel map that stores the necessary General Matrix Multiplication (GEMM) 
operations to be executed (Map step), and then use a Gather-GEMM-Scatter 
process to execute these GEMM operations (GMaS step).

In this work, we analyze the shortcomings of prior state-of-the-art SC engines, 
and propose Minuet, a novel memory-efficient SC engine tailored for modern GPUs,
where we
* Replace the hash tables used in the Map step with a novel segmented sorting 
double-traversed binary search algorithm that highly utilizes the on-chip memory
hierarchy of GPUs;
* Use a lightweight scheme to autotune the tile size in the Gather and Scatter 
operations of the GMaS step, such that to adapt the execution to the particular 
characteristics of each SC layer, dataset, and GPU architecture;
* Employ a padding-efficient GEMM grouping approach that reduces both memory 
padding and kernel launching overheads. 

Our evaluations show that Minuet significantly outperforms prior 
SC engines by on average $1.74\times$ (up to $2.22\times$) for end-to-end point 
cloud network executions. Our novel segmented sorting double-traversed binary 
search algorithm achieves superior speedups by $15.8\times$ on average 
(up to $26.8\times$) over prior SC engines in the Map step.

## Installation

```shell
pip3 install "torch~=2.1" "packaging~=23.2"
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip3 install .
```

## License

Please refer to the [LICENSE](LICENSE) file.
