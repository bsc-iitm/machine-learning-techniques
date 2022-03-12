# Loss

The loss depends on the nature of the problem being solved.



## Regression

$\boldsymbol{y}$ is a vector of target label for $n$ examples. $\boldsymbol{\hat{y}}$ is the output of the network and corresponds to the predicted labels. Both vectors are of size $n$. The loss is our usual squared error:
$$
L(\boldsymbol{y}, \boldsymbol{\hat{y}}) = \cfrac{1}{2} \cdot (\boldsymbol{\hat{y}} - \boldsymbol{y})^T (\boldsymbol{\hat{y}} - \boldsymbol{y})
$$


## Multi-class classification

$\boldsymbol{Y}$ is a matrix of target labels for $n$ examples. Each row of this matrix is a one-hot vector. $\boldsymbol{\hat{Y}}$ is a matrix of predicted probabilities. Both matrices are of size $n \times k$. The categorical cross-entropy loss is given as follows:
$$
L(\boldsymbol{Y}, \boldsymbol{\hat{Y}})=-\boldsymbol{1}^T_{n} \left( \boldsymbol{Y} \odot \log \boldsymbol{\hat{Y}} \right) \boldsymbol{1}_{k}
$$
This equation can be understood as follows:

- $\boldsymbol{1}_n$ and $\boldsymbol{1}_k$  are vectors of ones of sizes $n$ and $k$ respectively.
- If $\boldsymbol{M}$ is a matrix of size $n \times k$, then $\boldsymbol{1}^T_{n} \boldsymbol{M} \boldsymbol{1}_k$ is the sum of all elements in the matrix.
- $\odot$ is the element-wise product.



The `NumPy` equivalent of the loss is:



```python
L = -np.sum(Y * np.log(Y_hat))
```



Note that the loss function is always a scalar. To understand why the cross-entropy takes this form, refer to the appendix.

