---
title: Support Vector Machines
subtitle: Machine Learning Techniques
author: Karthik Thiagarajan
---

# Course Outline

- Linear regression

- Least square classification
- Perceptron
- Logistic regression
- Naive Bayes
- Softmax regression
- **Support Vector Machines (SVM)**
- Decision trees
- Ensemble techniques
- K-means clustering
- Artificial Neural Networks



# Lecture Outline

::: incremental

- Motivation
- Geometry
- **Hard-margin SVM**
  - Formulation
  - Optimization (recap)
  - **Optimization**
- Soft-margin SVM
- Approximate solution

:::



# Hard-margin SVM

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\min \limits_{w, b} \quad \cfrac{||w||^2}{2}
$$
<br>

subject to:

<br>
$$
y_i(w^Tx_i + b) \geq 1,\quad 1 \leq i \leq n
$$


:::

::: {.column width="50%"}

:::

:::



# Hard-margin SVM

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\min \limits_{w, b} \quad \cfrac{||w||^2}{2}
$$
<br>

subject to:

<br>
$$
1 - y_i(w^Tx_i + b) \leq 0,\quad 1 \leq i \leq n
$$


:::

::: {.column width="50%"}

:::

:::



# Step-1: Lagrangian

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
L(w, b, \lambda) = \cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i \left[ 1 - y_i(w^Tx_i + b)\right]
$$
<br>

<br>

:::

::: {.column width="50%"}

:::

:::



# Step-2: $\nabla L = 0$

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\nabla_w L = w - \sum \limits_{i = 1}^{n} \lambda_i y_i x_i = 0
$$
<br>

<br>

:::

::: {.column width="50%"}

:::

:::



# Step-2: $\nabla L = 0$

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
w = \sum \limits_{i = 1}^{n} \lambda_i y_i x_i
$$
<br>

<br>

:::

::: {.column width="50%"}

:::

:::



# Step-2: $\nabla L = 0$

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\nabla_b L =- \sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$
<br>

<br>

:::

::: {.column width="50%"}

:::

:::



# Step-2: $\nabla L = 0$

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$
<br>

<br>

:::

::: {.column width="50%"}

:::

:::



# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
L(w, b, \lambda) = \cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i \left[ 1 - y_i(w^Tx_i + b)\right]
$$
<br>
$$
w = \sum \limits_{1 = 1}^{n} \lambda_i y_i x_i
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

:::

:::



# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
L(w, b, \lambda) = \cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i -\sum \limits_{i = 1}^{n} \lambda_i y_i(w^Tx_i) + \sum \limits_{i = 1}^{n} \lambda_i y_i b
$$
<br>
$$
w = \sum \limits_{1 = 1}^{n} \lambda_i y_i x_i
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

:::

:::



# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
L(w, b, \lambda) = \cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i -\sum \limits_{i = 1}^{n} \lambda_i y_i(w^Tx_i)
$$
<br>
$$
w = \sum \limits_{1 = 1}^{n} \lambda_i y_i x_i
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

:::

:::



# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
L(w, b, \lambda) = \cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i -\sum \limits_{i = 1}^{n} \lambda_i y_i(w^Tx_i)
$$
<br>
$$
w = \sum \limits_{1 = 1}^{n} \lambda_i y_i x_i
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

<br>
$$
\begin{aligned}
\sum \limits_{i = 1}^{n} \lambda_i y_i (w^T x_i) &= w^T \left (\sum \limits_{i = 1}^{n} \lambda_i y_i x_i  \right)\\\\
&= w^T w\\\\
&= ||w||^2
\end{aligned}
$$


:::

:::



# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
L(w, b, \lambda) = -\cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i
$$
<br>
$$
w = \sum \limits_{1 = 1}^{n} \lambda_i y_i x_i
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

<br>



:::

:::







# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
L(w, b, \lambda) = -\cfrac{||w||^2}{2} + \sum \limits_{i = 1}^{n} \lambda_i
$$
<br>
$$
w = \sum \limits_{1 = 1}^{n} \lambda_i y_i x_i
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

<br>
$$
\begin{aligned}
||w||^2 &= w^Tw\\
&= \left ( \sum \limits_{i = 1}^{n} \lambda_i y_i x_i \right)^T \left( \sum \limits_{j = 1}^{n} \lambda_j y_j x_j \right)\\
&= \left ( \sum \limits_{i = 1}^{n} \lambda_i y_i x_i^T \right) \left( \sum \limits_{j = 1}^{n} \lambda_j y_j x_j \right)\\
&= \cfrac{1}{2} \sum \limits_{i = 1}^{n} \sum \limits_{j = 1}^{n} (y_i y_j x_i^Tx_j)  \lambda_i \lambda_j
\end{aligned}
$$


:::

:::



# Step-3: Eliminate $w, b$

::: {.columns align=center}

::: {.column width="100%"}

<br>
$$
L(\lambda) = \sum \limits_{i = 1}^{n} \lambda_i - \cfrac{1}{2} \sum \limits_{i = 1}^{n} \sum \limits_{j = 1}^{n} (y_i y_j x_i^Tx_j)  \lambda_i \lambda_j
$$

:::

::: {.column width="50%"}

<br>


:::

:::



# Step-4: Dual

::: {.columns align=center}

::: {.column width="100%"}

<br>
$$
\max \limits_{\lambda}\quad  \sum \limits_{i = 1}^{n} \lambda_i - \cfrac{1}{2} \sum \limits_{i = 1}^{n} \sum \limits_{j = 1}^{n} (y_i y_j x_i^Tx_j)  \lambda_i \lambda_j
$$
<br>

subject to the constraints:
$$
\lambda_i \geq 0, \quad 1 \leq i \leq n
$$
and
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

<br>


:::

:::



# Step-5: Quadratic programming

::: {.columns align=center}

::: {.column width="50%"}

<br>
$$
\max \limits_{\lambda}\quad  \sum \limits_{i = 1}^{n} \lambda_i - \cfrac{1}{2} \sum \limits_{i = 1}^{n} \sum \limits_{j = 1}^{n} (y_i y_j x_i^Tx_j)  \lambda_i \lambda_j
$$
<br>

subject to the constraints:
$$
\lambda_i \geq 0, \quad 1 \leq i \leq n
$$
and
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

<br>

<br>

<br>

QP solver returns optimal $\lambda$

:::

:::



# Step-6: Compute $w$ and $b$

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>

<br>
$$
\large w = \sum \limits_{i = 1}^{n} \lambda_i y_i x_i
$$
:::

::: {.column width="50%"}

:::

:::



# Support vectors

::: {.columns align=left}

::: {.column width="40%"}

<br>

<br>

<br>
$$
\large w = \sum \limits_{i = 1}^{n} \lambda_i y_i x_i
$$
:::

::: {.column width="60%"}

::: incremental

- KKT condition: $\lambda_i \geq 0$
- KKT condition: $\lambda_i \left[1 - y_i(w^T x_i + b) \right] = 0$
- If constraint is active, $y_i(w^T x_i + b) = 1$ and $\lambda_i > 0$
- If constraint is inactive, $y_i(w^Tx_i + b) > 1$ and $\lambda_i = 0$
- For most of the data-points $\lambda_i = 0$
- Those points for which $\lambda_i > 0$ are the support vectors
- Support vectors lie on lines that are parallel to the decision boundary:
  - $w^T x + b = 1$
  - $w^T x + b = -1$

- If $S$ is the set of support vectors, then we can rewrite $w$ as:
- $w = \sum \limits_{x_i \in S} \lambda_i y_i x_i$

:::

:::

:::



# Support vectors

::: {.columns align=left}

::: {.column width="40%"}

<br>

<br>

<br>
$$
\large w = \sum \limits_{x_i \in S} \lambda_i y_i x_i
$$
:::

::: {.column width="60%"}

![](images/021.svg){width="800"}

:::

:::



# Inference

::: {.columns align=left}

::: {.column width="100%"}

<br>

<br>

<br>
$$
\hat{y} = \text{sign}(w^Tx + b)
$$
:::

::: {.column width="50%"}

:::

:::

