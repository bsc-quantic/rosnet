# Welcome to rosnet's documentation!

<center style="font-family: monospace">
rosnet | ɟɘnƨoɿ
</center>

`rosnet` is a Python library for distributed execution of tensor operations. It was designed for simulation of quantum circuits and tensor networks, but it may be used as a generic array backend.

```{warning}
`rosnet` is under heavy development and its API is not fully stabilized yet. Breaking changes may happen in the future.
```

```{toctree}
:hidden:
install
integration
examples/index
api
develop/index
```

## Citing
`rosnet` is published in the proceedings of the 2021 IEEE/ACM Second International Workshop on Quantum Computing Software (QCS) [here](https://ieeexplore.ieee.org/abstract/document/9651410), which took place in the Supercomputing'21 conference.
If you don't have access to the paper, the [preprint](https://arxiv.org/abs/2201.06620) is available in ArXiv.
If it's useful in your research, please consider citing it!

```{code-block} latex
@INPROCEEDINGS{rosnet,
   author={Sánchez-Ramírez, Sergio and Conejero, Javier and Lordan, Francesc and Queralt, Anna and Cortes, Toni and Badia, Rosa M and García-Saez, Artur},
   booktitle={2021 IEEE/ACM Second International Workshop on Quantum Computing Software (QCS)},
   title={RosneT: A Block Tensor Algebra Library for Out-of-Core Quantum Computing Simulation},   year={2021},
   volume={},
   number={},
   pages={1-8},
   doi={10.1109/QCS54837.2021.00004}
}
```


## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
