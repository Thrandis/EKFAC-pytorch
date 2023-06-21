# EKFAC and K-FAC Preconditioners for Pytorch
This repo contains a Pytorch implementation of the EKFAC and K-FAC preconditioners. If you find this software useful, please check the references below and cite accordingly!

### Fork Note

This fork's sole purpuse is to update the code to work with the latest Pytorch version removing deprecated functions.

### Presentation

We implemented K-FAC and EKFAC as `preconditioners`. Preconditioners are similar Pytorch's `optimizer` class, with the exception that they do not perform the update of the parameters, but only change the gradient of those parameters. They can thus be used in combination with your favorite optimizer (we used SGD in our experiments). Note that we only implemented them for `Linear` and `Conv2d` modules, so they will silently skip all the other modules of your network.

### Usage

Here is a simple example showing how to add K-FAC or EKFAC to your code:

```python
# 1. Instantiate the preconditioner
preconditioner = EKFAC(network, 0.1, update_freq=100)

# 2. During the training loop, simply call preconditioner.step() before optimizer.step().
#    The optimiser is usually SGD.
for i, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = network(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    preconditioner.step()  # Add a step of preconditioner before the optimizer step.
    optimizer.step()
```

### References

#### EKFAC:
 - Thomas George, César Laurent, Xavier Bouthillier, Nicolas Ballas, Pascal Vincent, _[Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis](https://arxiv.org/abs/1806.03884)_, NIPS 2018

#### K-FAC:
- James Martens, Roger Grosse, _[Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671)_, ICML 2015
#### K-FAC for Convolutions:
- Roger Grosse, James Martens, _[A Kronecker-factored Approximate Fisher Matrix for Convolution Layers](https://arxiv.org/abs/1602.01407)_, ICML 2016
- César Laurent, Thomas George, Xavier Bouthillier, Nicolas Ballas, Pascal Vincent, _[An Evaluation of Fisher Approximations Beyond Kronecker Factorization](https://openreview.net/pdf?id=ryVC6tkwG)_, ICLR Workshop 2018
#### Norm Constraint:
- Jimmy Ba, Roger Grosse, James Martens, _[Distributed Second-order Optimization using Kronecker-Factored Approximations](https://jimmylba.github.io/papers/nsync.pdf)_, ICLR 2017
- Jean Lafond, Nicolas Vasilache, Léon Bottou, _[Diagonal Rescaling For Neural Networks](https://arxiv.org/abs/1705.09319)_, arXiv 2017
