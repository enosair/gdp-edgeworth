# gdp-edgeworth

This repo contains simple demo code for the following paper, to compute the composition bound of Gaussian differential privacy using the degree-2 Edgeworth approximation.

> Qinqing Zheng, Jinshuo Dong, Qi Long, Weijie J. Su. *Sharp Composition Bounds for Gaussian Differential Privacy via Edgeworth Expansion*. ICML 2020. [[arXiv:2003.04493](https://arxiv.org/abs/2003.04493)].

### Introduction
Datasets containing sensitive information are often sequentially analyzed by many
algorithms. This raises a fundamental question in differential privacy is concerned with
how the overall privacy bound degrades under composition.  To address this question, we
introduce a family of analytical and sharp privacy bounds under composition using the
Edgeworth expansion in the framework of the recently proposed f-differential privacy.
In contrast to the existing composition theorems using the central limit theorem, our
new privacy bounds under composition gain improved tightness by leveraging the refined
approximation accuracy of the Edgeworth expansion. Our approach leverages the hypothesis
testing interpretation of differential privacy, is easy to implement and computationally
efficient for any number of compositions.


### Code organization
- `edgeworth.py`: math functions and utilities to compute the Edgeworth approximation, CLT estimation, and the numerical computation of the exact composition bound.
- `expr_laplace.py`: main interface to run experiments to compare the three methods for testing $\texttt{Lap}(0, 1)^{\otimes n}$ vs $\texttt{Lap}(3/\sqrt{n}, 1)^{\otimes n}$.
- `expr_mixture.py`: main interface to run experiments to compare the three methods for testing the standard normal distribution $\N(0, 1)$ vs the mixture model $p\N(\frac1\sigma, 1) + (1-p)\N(0,1)$. This is the privacy guarantee (before symmetrization) of models trained by $n$-step noisy SGD.
- `edgeworth.ipynb`: Jupyter notebook that contains example usage of our code.
