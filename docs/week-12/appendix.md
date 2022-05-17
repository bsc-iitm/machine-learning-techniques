# Appendix

## Categorical Cross Entropy

### Part-1

In the case of multi-class classification, the output layer after softmax can be viewed as a conditional probability distribution over the labels given the data-point. This is a categorical distribution over the $k$ classes. Let us call it $q(y\ |\ x)$. This is the predicted distribution. There is a true distribution over the labels. Let us call it $p(y\ |\ x)$. Note that both are conditional distributions of the label given the data-point.

The (conditional) cross-entropy of these two conditional distributions is given as follows:
$$
H \Big( p(y\ |\ x), q(y\ |\ x) \Big) = -\sum \limits_{x} p(x) \sum \limits_{y} p(y\ |\ x) \log q(y\ |\ x)
$$
Note that this is really the expectation of some function $f(x)$ over the marginal distribution $p(x)$: 


$$
\begin{aligned}
E[f(x)] &= \sum \limits_{x} p(x) f(x)\\
&= E \left[- \sum \limits_{y} p(y\ |\ x) \log q(y\ |\ x) \right]
\end{aligned}
$$


To compute an expectation, we need to sum over all samples drawn from this distribution. In practice, this is not going to be possible. This expectation is typically approximated by sampling data-points from the marginal distribution. We can replace the exhaustive summation with a sum over the $n$ data-points in our batch of examples:
$$
H \Big( p(y\ |\ x), q(y\ |\ x) \Big) = -\cfrac{1}{n} \sum \limits_{i = 1}^{n} \sum \limits_{y} p(y\ |\ x^{(i)}) \log q(y\ |\ x^{(i)})
$$
A crude way of seeing this is as follows: if a particular data-point $x^{*}$ is sampled $r$ times, (appears $r$ times in the batch) then:
$$
p(x^{*}) \approx \cfrac{r}{n}
$$
Let us get rid of the factor of $\frac{1}{n}$ as it doesn't matter from the point of view of optimization.

### Part-2

We are left with the following expression for the cross-entropy:
$$
H \Big( p(y\ |\ x), q(y\ |\ x) \Big) = -\sum \limits_{i = 1}^{n} \sum \limits_{y} p(y\ |\ x^{(i)}) \log q(y\ |\ x^{(i)})
$$
We now need to understand the inner sum over the labels. Some notation needs to be be setup to make things more tractable:


$$
\begin{aligned}
y^{(i)}&: \text{true label for data-point i, scalar}\\
\boldsymbol{y^{(i)}}&: \text{true label for data-point i, one-hot vector}\\
\boldsymbol{\hat{y}^{(i)}}&: \text{predicted probabilities for data-point i, vector}
\end{aligned}
$$


The symbols in bold are are vectors of length $k$, where $k$ is the number of classes. With this notation we can get a better idea about the true and predicted distributions, $p$ and $q$, respectively:


$$
\begin{aligned}
p(y = c\ |\ x^{(i)}) &= \boldsymbol{y^{(i)}_{c}}\\
q(y = c\ |\ x^{(i)}) &= \boldsymbol{\hat{y}^{(i)}_{c}}
\end{aligned}
$$


Note that $\boldsymbol{\hat{y}^{(i)}_{c}}$ is the $c^{th}$ component of a vector and hence a scalar. Same is the case for $\boldsymbol{y^{(i)}_{c}}$  We are now ready to compute the categorical cross-entropy:


$$
\begin{aligned}
H \Big( p(y\ |\ x), q(y\ |\ x) \Big) &= -\sum \limits_{i = 1}^{n} \sum \limits_{y} p(y\ |\ x^{(i)}) \log q(y\ |\ x^{(i)})\\
&= - \sum \limits_{i = 1}^{n} \sum \limits_{c = 1}^{k} p(y = c\ |\ x^{(i)}) \log q(y = c\ |\ x^{(i)})\\
&= - \sum \limits_{i = 1}^{n} \sum \limits_{c = 1}^{k} \boldsymbol{y^{(i)}_{c}}\ \cdot \log(\boldsymbol{\hat{y}^{(i)}_{c}})
\end{aligned}
$$


This is the final expression for the categorical cross-entropy (CCE) loss. Even though the inner sum is over $k$ classes, only one of the values for $\boldsymbol{y^{(i)}_{c}}$ is non-zero. In the end, the sum reduces to a very simple scalar expression in principle:


$$
\text{CCE} = \sum \limits_{i = 1}^{n} -\log \boldsymbol{\hat{y}^{(i)}_{y^{(i)}}}
$$



I know that this looks hopelessly complicated, thanks to the subscripts and superscripts. But what it is saying is this:

The categorical cross-entropy for a batch of data-points can be computed as follows:

- For each data-point, get the predicted probability corresponding to the true label.
- Take negative log of that probability.
- Sum this quantity across the entire batch.



### Part-3

Intuitively, this loss makes a lot of sense. In fact, we can see why this should qualify as a loss in the first place by asking these questions:

- When is this quantity minimized?



When the predicted probability for the correct class (true label) is $1$ for every data-point in the batch. 

- What happens if the predicted probability for the correct class for some data-point is close to $0$? 



In such a case, $-\log \boldsymbol{\hat{y}^{(i)}}_{y^{(i)}}$  will be a huge positive value, something that is undesirable. 

- What does minimizing cross-entropy really mean?



The attempt to minimize the categorical cross-entropy translates to pushing the probability of the correct-class closer and closer to $1$ for each data-point. Our hope is that in this process, the model will learn something useful that generalizes to unseen data-points as well.



### Part-4

We can see how this reduces to the binary cross-entropy loss in the case of two classes:


$$
\begin{aligned}
BCE &= - \sum \limits_{i = 1}^{n} \sum \limits_{c = 1}^{2} \boldsymbol{y^{(i)}_{c}}\ \cdot \log(\boldsymbol{\hat{y}^{(i)}_{c}})\\
&= - \sum \limits_{i = 1}^{n} \left(  y^{(i)} \log (\hat{y}^{(i)}) + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right)
\end{aligned}
$$



## Chain rule for matrix product

Let $P = QR$, where $P, Q$ and $R$ are matrices of compatible dimensions. Let $\phi$ be some function of $P$. Let us assume that we have access to the gradient of $\phi$ with respect to $P$. Call this $P^{(g)}$. Then, we have the following equations:


$$
Q^{(g)} = P^{(g)} R^T\\
R^{(g)} = Q^T P^{(g)}
$$


This is nothing but the chain rule of differentiation in the presence of a matrix product.



## Gradient for softmax layer

In order to derive the gradient matrix at the final layer, it is easier to consider the simpler case of the gradient vector. Let $\boldsymbol{z}$ be the pre-activation vector at layer $L$ of the network. After applying softmax function $g$ on the pre-activation vector, we get $\boldsymbol{a}$, the activation vector. The two vectors are related by the following equations:


$$
\boldsymbol{a} = g(\boldsymbol{z}) = \begin{bmatrix}
\cdots & \cfrac{e^{z_j}}{\sum \limits_{r = 1}^{k} e^{z_r}} & \cdots \end{bmatrix}
$$

The gradient of the loss with respect to the activation vector is given by:



$$
\boldsymbol{a^{(g)}} = \nabla_{\boldsymbol{a}} L
$$



The tricky part is the gradient of the loss with respect to the pre-activations. Let us consider a single element in the pre-activation vector, $z_j$. We are interested in the partial derivative $\cfrac{\partial L}{\partial z_j}$. What this means is this:  how much does the loss change when a small change is made to $z_j$. It is not exactly this, but the ratio of these two quantities, but you get the idea.



$z_j$ isn't directly connected to the loss function. The elements of the activation vector $a_i$s intervene. In other words, the impact of $z_j$ on the loss is mediated by the $a_i$s. Changing $z_j$ has an impact on each of the $a_i$s because of the way softmax is defined. We already know how the loss is affected when the $a_j$s are disturbed. This is captured by $\boldsymbol{a^{(g)}}$. So much for the chain rule. We are now ready to state it:



$$
\cfrac{\partial L}{\partial z_j} = \sum \limits_{i = 1}^{k} \cfrac{\partial L}{\partial a_i} \cfrac{\partial a_i}{\partial z_j}
$$



The second term in the product is the $ij^{th}$ element of the Jacobian matrix $J_{\boldsymbol{z}}(\boldsymbol{a})$:


$$
J_{\boldsymbol{z}}(\boldsymbol{a}) \Bigg|_{ij} = \left [ \cfrac{\partial a_i}{\partial z_j} \right]
$$


The Jacobian matrix is a matrix of partial derivatives of the components of the vector $\boldsymbol{a}$ with respect to $\boldsymbol{z}$. With the introduction of the Jacobian matrix, we can now state the gradient of the loss with respect to the pre-activations in the following succinct manner:



$$
\boldsymbol{z^{(g)}} = \nabla_{\boldsymbol{z}} L = J_{\boldsymbol{z}}(\boldsymbol{a})^T\boldsymbol{a^{(g)}}
$$



We can now turn our attention to computing the Jacobian:



$$
J_{\boldsymbol{z}}(\boldsymbol{a}) \Bigg|_{ij} = \begin{cases}
-a_i a_j &\quad i \neq j\\
a_j - a_j^2 &\quad i = j
\end{cases}
$$



This form of expressing the Jacobian element-wise is unwieldy for computation. Fortunately, this has a simple matrix-representation:

$$
J_{\boldsymbol{z}}(\boldsymbol{a}) = \text{diag}(\boldsymbol{a}) - \boldsymbol{a} \boldsymbol{a}^T
$$

Here, $\text{diag}({\boldsymbol{a}})$ is a diagonal matrix with the elements of the activation vector making up the diagonal. The second term on the RHS is called the outer-product. We can now plug the Jacobian into the earlier equation:



$$
\begin{aligned}
\boldsymbol{z^{(g)}} &= J_{\boldsymbol{z}}(\boldsymbol{a})^T\boldsymbol{a^{(g)}}\\
&= \Big[ \text{diag}(\boldsymbol{a}) - \boldsymbol{a} \boldsymbol{a}^T \Big] \boldsymbol{a^{(g)}}\\
&= \left( \boldsymbol{a} \odot \boldsymbol{a^{(g)}} \right) - \left( \boldsymbol{a}^T \boldsymbol{a^{(g)}}\boldsymbol{a}\right)\\
&= \left(\boldsymbol{a} \odot \boldsymbol{a^{(g)}} \right) - \left( \boldsymbol{a} \odot \boldsymbol{a^{(g)}} \boldsymbol{1}_k \boldsymbol{a}\right)\\
\end{aligned}
$$



That might seem terribly complicated. Step-2 of the equation is just substituting the Jacobian. To understand step-3, try to think about the product of a diagonal matrix and a vector, and see how that transforms to element-wise product between two vectors. The final-step is another trick where we convert the dot product between two vectors into an element-wise product followed by multiplication by a vector of ones.

Why are all these transformations necessary? The moment we express them as element-wise products, we can almost effortlessly extend this formula to a matrix of activations:



$$
\boldsymbol{Z_L^{(g)}} = \left(\boldsymbol{A_L} \odot \boldsymbol{A_L^{(g)}} \right) - \left( \boldsymbol{A_L} \odot \boldsymbol{A_L^{(g)}} \boldsymbol{1}_{k \times k} \right) \odot \boldsymbol{A_L}\\
$$



That again looks complicated. But if we stare at it for a while, the RHS is actually remarkably simple:

- The first term is is simply the element-wise multiplication of the activation matrix and its corresponding gradient. Call this some matrix $\boldsymbol{B}$.
- The second term is just the row-wise sum of $\boldsymbol{B}$ followed by an element-wise multiplication with the activation-matrix.

These expressions are powerful because we can directly translate them into `NumPy`. For example, the `NumPy` equivalent of this equation would be:

```python
B = A_L * A_L_g
Z_l_g = B - np.sum(B, axis = 1) * A_l
```

The good news is that the expression for $\boldsymbol{Z_L^{(g)}}$ In the case of softmax with categorical cross-entropy loss can be further simplified. Recall the following results:



$$
\begin{aligned}
\boldsymbol{A_L} &= \boldsymbol{\hat{Y}}\\
\boldsymbol{A_L^{(g)}} &= - \boldsymbol{Y} \odot \boldsymbol{\hat{Y}}^{\odot -1} 
\end{aligned}
$$



The element-wise product of these two matrices is simply the matrix $-\boldsymbol{Y}$! With that, the expression for $\boldsymbol{Z_L^{(g)}}$ becomes:

$$
\boldsymbol{Z_L^{(g)}} = \boldsymbol{\hat{Y}} - \boldsymbol{Y}
$$



All this computation seems justified given the elementary nature of the result.

