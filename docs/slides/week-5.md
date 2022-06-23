---
title: MLT | Week-5
---

# Course Outline

- Linear regression

- Least square classification
- Perceptron
- Logistic regression
- Naive Bayes
- Softmax regression
- Support Vector Machines (SVM)
- Decision trees
- Ensemble techniques
- K-means clustering
- Artificial Neural Networks



# Story so far

**Regression**

::: incremental

- Linear regression
- Polynomial regression

:::



# Story so far

**Regression**

- Linear regression
- Polynomial regression

<br>

**Classification**

::: incremental

- Least square classification
- Perceptron

:::



# Unifying theme

::: {.columns align=center}

::: {.column width="100%"}

:::

<br>

<br>

What is one striking similarity among all the models we have seen so far?

::: {.column width="0%"}

:::

:::



# Linearity

::: {.columns align=center}

::: {.column width="100%"}

:::

<br>

<br>

::: {.column width="0%"}

:::

:::



# Linearity

::: {.columns align=center}

::: {.column width="100%"}

:::

<br>

<br>
$$
\huge{w^T x}
$$
:::

::: {.column width="0%"}



:::







# Linearity

::: {.columns align=center}

::: {.column width="100%"}

:::

<br>

<br>
$$
\huge{w^T x} = w_0 + w_1x_1 + \cdots + w_n x_n
$$
:::

::: {.column width="0%"}



:::



# Linearity

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
w^T x = w_0 + w_1x_1 + \cdots + w_n x_n
$$
:::

::: {.column width="50%"}

<br>

<br>

What does $w_i$ represent?

:::

:::



# Linearity

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
w^T x = w_0 + w_1x_1 + \cdots + w_n x_n
$$
:::

::: {.column width="50%"}

<br>

<br>

What does $w_i$ represent?

<br>

The importance of feature $x_i$

:::

:::



# Linearity

::: {.columns align=left}

::: {.column width="50%"}

<br>

<br>
$$
w^T x
$$
:::

::: {.column width="50%"}

<br><br>

::: incremental

- Linear regression
- Polynomial regression
- Least square classification
- Perceptron
- Logistic Regression?

:::

:::

:::



# Logistic Regression

::: {.columns align=left}

::: {.column width="100%"}

<br>

<br>

**Setting**

::: incremental

- Binary classification
- Discriminative model

:::

:::

::: {.column width="0%"}

:::

:::



# Task

::: {.columns align=left}

::: {.column width="100%"}

<br>

::: incremental

- Feature matrix: $X$
  - $m \times n$
  - $m$ data-points, $n$ features
- Label vector: $y$
  - $m \times 1$
  - $0$ or $1$

:::

:::

::: {.column width="0%"}

:::

:::



# Task

::: {.columns align=left}

::: {.column width="50%"}

<br>

- Feature matrix: $X$
  - $m \times n$
  - $m$ data-points, $n$ features
- Label vector: $y$
  - $m \times 1$
  - $0$ or $1$

:::

::: {.column width="50%"}

<br>

Learn a function $h$ such that given a feature vector $x$, the predicted label is given as:
$$
y_{\text{pred}} = h(x)
$$


:::

:::





# Types of models

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
P(y\ |\ x)
$$
<br>

Discriminative

:::

::: {.column width="50%"}

<br>

<br>
$$
P(x, y)
$$
<br>

Generative

:::

:::



# Logistic Regression

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
P(y = 1\ |\ x) = ?
$$


:::

::: {.column width="0%"}

:::

:::



# Logistic Regression

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
P(y = 1\ |\ x) = \sigma(w^T x)
$$


:::

::: {.column width="0%"}

:::

:::



# Logistic Regression

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
P(y = 1\ |\ x) = \sigma(w^T x)
$$


<br>

What kind of a random variable is $y$?

:::

::: {.column width="0%"}

:::

:::



# Sigmoid

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\sigma(z) = \cfrac{1}{1 + e^{-z}}
$$


:::

::: {.column width="0%"}

:::

:::



# Sigmoid

::: {.columns align=center}

::: {.column width="20%"}

<br>

<br>
$$
\sigma(z) = \cfrac{1}{1 + e^{-z}}
$$


:::

::: {.column width="80%"}

Here $g = \sigma$:

![](../assets/images/img_8.svg){width="500"}

:::

:::



# Logistic Regression

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
P(y = 1\ |\ x) = \sigma(w^T x)
$$


:::

::: {.column width="50%"}

<br>

<br>
$$
P(y = 0\ |\ x) = ?
$$


:::

:::



# Logistic Regression

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
P(y = 1\ |\ x) = \sigma(w^T x)
$$


:::

::: {.column width="50%"}

<br>

<br>
$$
P(y = 0\ |\ x) = 1 - \sigma(w^T x)
$$


:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}
$$
\begin{aligned}
L(w) &= \prod \limits_{i = 1}^{m} P(y = y_i\ |\ x_i)\quad \quad \quad \quad \quad \quad\ \ \ \\\\\
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}
$$
\begin{aligned}
L(w) &= \prod \limits_{i = 1}^{m} P(y = y_i\ |\ x_i)\\\\
&= \prod \limits_{i = 1}^{m} \left[ \sigma(w^Tx_i) \right]^{y_i} \left[ 1 - \sigma(w^Tx_i) \right]^{1 - y_i}
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}
$$
\begin{aligned}
L(w) &= \prod \limits_{i = 1}^{m} \left[ \sigma(w^Tx_i) \right]^{y_i} \left[ 1 - \sigma(w^Tx_i) \right]^{1 - y_i}
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}
$$
\begin{aligned}
l(w) &= \log (L(w))\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \\\\
&= \log \left (\prod \limits_{i = 1}^{m} \left[ \sigma(w^Tx_i) \right]^{y_i} \left[ 1 - \sigma(w^Tx_i) \right]^{1 - y_i} \right)
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}
$$
\begin{aligned}
l(w) &= \log (L(w))\\\\
&= \log \left (\prod \limits_{i = 1}^{m} \left[ \sigma(w^Tx_i) \right]^{y_i} \left[ 1 - \sigma(w^Tx_i) \right]^{1 - y_i} \right)\\\\
&= \sum \limits_{i = 1}^{m} y_i \cdot \log (\sigma(w^T x_i)) + (1 - y_i) \log (1 - \sigma(w^Tx_i))
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\begin{aligned}
\max\quad  l(w)
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\begin{aligned}
\min\quad  -l(w)
\end{aligned}
$$


:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\begin{aligned}
\min\quad  -l(w)
\end{aligned}
$$


<br>

Minimize the negative log-likelihood

:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="100%"}

<br>

<br>
$$
\begin{aligned}
\min\quad - \sum \limits_{i = 1}^{m} y_i \cdot \log (\sigma(w^T x_i)) + (1 - y_i) \log (1 - \sigma(w^Tx_i))
\end{aligned}
$$


<br>

Note: $-$ sign is for the entire sum

:::

::: {.column width="0%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
\begin{aligned}
- \left[ y_i \cdot \log (\sigma(w^T x_i)) + (1 - y_i) \log (1 - \sigma(w^Tx_i))\right]
\end{aligned}
$$
<br>

:::

::: {.column width="50%"}

:::

:::



# MLE

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>
$$
\begin{aligned}
- \left[ y_i \cdot \log (\sigma(w^T x_i)) + (1 - y_i) \log (1 - \sigma(w^Tx_i))\right]
\end{aligned}
$$
<br>

:::

::: {.column width="50%"}

<br>

<br>
$$
-\left [ p \log q + (1 - p) \log (1 - q) \right]
$$


:::

:::