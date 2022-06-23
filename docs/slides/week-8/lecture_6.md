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
- Hard-margin SVM
  - Formulation
  - Optimization (recap)
  - Optimization
- **Soft-margin SVM**
- Approximate solution

:::



# Nearly Linearly separable

::: {.columns align=center}

::: {.column width="100%"}

![](images/034.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Nearly Linearly separable

::: {.columns align=center}

::: {.column width="100%"}

![](images/035.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/038.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/039.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/040.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/041.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/042.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/043.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/044.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/045.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::

# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/046.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="100%"}

![](images/047.svg){width="800"}

:::

::: {.column width="0%"}

:::

:::



# Errors

::: {.columns align=center}

::: {.column width="40%"}

![](images/048.svg){width="700"}

:::

::: {.column width="60%"}

<br>

<br>

<br>
$$
\xi_i = \begin{cases}
0,\ & x_i \text{ outside margin}\\
1 - y_i(w^T x_i + b), \ & x_i \text{ inside margin}\\
\end{cases}
$$


:::

:::



# Constraints

::: {.columns align=center}

::: {.column width="40%"}

![](images/048.svg){width="700"}

:::

::: {.column width="60%"}

<br>

<br>
$$
\xi_i = \begin{cases}
0,\ & x_i \text{ outside margin}\\
1 - y_i(w^T x_i + b), \ & x_i \text{ inside margin}\\
\end{cases}
$$
<br>
$$
\xi_i \geq 0
$$
and
$$
y_i(w^T x_i + b) \geq 1 - \xi_i
$$


:::

:::





# Objective function

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\cfrac{||w||^2}{2} + C \sum \limits_{i = 1}^{n} \xi_i
$$


:::

::: {.column width="50%"}

:::

:::



# Primal

::: {.columns align=center}

::: {.column width="100%"}

<br>
$$
\min \limits_{w, b} \quad\cfrac{||w||^2}{2} + C \sum \limits_{i = 1}^{n} \xi_i
$$
<br>

subject to

<br>
$$
\xi_i \geq 0,\quad 1 \leq i \leq n
$$
<br>
$$
y_i(w^T x_i + b) \geq 1 - \xi_i, \quad 1 \leq i \leq n
$$


:::

::: {.column width="50%"}

:::

:::



# Dual

::: {.columns align=center}

::: {.column width="100%"}

<br>
$$
\max \limits_{\lambda}\quad  \sum \limits_{i = 1}^{n} \lambda_i - \cfrac{1}{2} \sum \limits_{i = 1}^{n} \sum \limits_{j = 1}^{n} (y_i y_j x_i^Tx_j)  \lambda_i \lambda_j
$$
<br>

subject to:

<br>
$$
0 \leq \lambda_i \leq C, \quad 1 \leq i \leq n
$$
<br>
$$
\sum \limits_{i = 1}^{n} \lambda_i y_i = 0
$$


:::

::: {.column width="50%"}

:::

:::

