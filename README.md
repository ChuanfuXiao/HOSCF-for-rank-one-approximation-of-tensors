## Abstract
Finding the best rank-one approximation is one of the most fundamental tasks in tensor computation and is widely used in various applications, such as wireless communication and multi-target tracking. In order to fully exploit modern multi-core parallel computers, we develop a novel decoupling algorithm that is parallel friendly. Specifically, we first build a bridge between the rank-one approximation of tensors and the eigenvector-dependent nonlinear eigenvalue problem (NEPv) and then proposed the higher-order self-consistent field (HOSCF) algorithm. Numerical results demonstrate the advantages of the proposed HOSCF algorithm.

## Theorem
Mathematically, finding the best rank-one approximation of tensors is equivalent to solving the following NEPv:
![HOSCF Algorithm](/img/th1.png)

## Algorithm
We can directly use the SCF iteration to solve the rank-one approximation of tensors, i.e., the proposed HOSCF algorithm.
![HOSCF Algorithm](/img/algorithm.png)

Moreover, we establish a rigorous convrgende theory of HOSCF.
![HOSCF Algorithm](/img/thm2.png)

## Experiments
Part 1: Convergence

We compare the convergence of the proposed HOSCF algorithm with other decoupling algorithms, including Jacobi-HOPM, Jacobi-ASVD, and GRQI. The input tensor is randomly generated, and each element of it follows the Gaussian distribution. We set the tensor size to 10, and increase the order of tensors from d=3 to 6.

![HOSCF Algorithm](/img/convergence1.png)

![HOSCF Algorithm](/img/convergence2.png)


Part 2: Parallel scalability

In order to present the scalability of HOSCF, we implement the HOSCF and HOPM(based ALS) algorithm in C++ language with OpenMP multi-threading and MPI multi-processes and the involved linear algebra operations such as eigensolver available from the Intel MKL library. The figure below illustrates the running time and speedup of the HOPM and HOSCF algorithms on 5th, 6th, 7th, and 8th-order (d) tensor as the number of processor cores increases from 1 to 256 (on 8 compute nodes, each node equipped with an Intel Xeon Platinum 8358P CPU of 2.60 GHz).

![Figure](/img/scalability.png)


## Quick Start
The HOSCF depends on blas and lapack. A convenient way to get this ready is [oneapi](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

```
1. install oneAPI Base and hpc Toolkit
2. source setvars.sh(perhaps at /opt/intel/oneapi/setvars.sh)
```


### Compile and Run
```
1. mkdir build
2. cmake ..
3. make -j 9
4. ./run -t 8 -n 6 -s 16 -r 10
```

### Result
```
./run -t 8 -n 6 -s 16 -r 10
[MPI] World size 1
[CONFIG] Number dim 6
[CONFIG] Shape 16
[Time] Avg time 1639.852300 ms
[Time] eigenpair J 46.879700 ms
Proportion 0.0285878
[Time] create J 1590.469900 ms
Proportion 0.969886
```

## Citation
If you are interested our work, please find the detailed in [paper](https://arxiv.org/abs/2403.01778)

If you use HOSCF in your research, please cite our paper:
```
@misc{xiao2024hoscf,
      title={HOSCF: Efficient decoupling algorithms for finding the best rank-one approximation of higher-order tensors}, 
      author={Chuanfu Xiao and Zeyu Li and Chao Yang},
      year={2024},
      eprint={2403.01778},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```
