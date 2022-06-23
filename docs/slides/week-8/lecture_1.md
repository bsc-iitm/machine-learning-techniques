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

- **Motivation**
- Geometry
- Hard-margin SVM
  - Formulation
  - Optimization (recap)
  - Optimization
- Soft-margin SVM
- Approximate solution


:::



# Story so far

::: {.columns align=center}

::: {.column width="100%"}

![](images/001.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# Story so far

::: {.columns align=center}

::: {.column width="100%"}

![](images/002.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# Story so far

::: {.columns align=left}

::: {.column width="50%"}

![](images/002.svg)

:::

::: {.column width="50%"}

<br>

<br>

::: incremental

- Least square classification
- Perceptron
- Logistic regression

:::

:::

:::



# Story so far

::: {.columns align=left}

::: {.column width="50%"}

![](images/002.svg)

:::

::: {.column width="50%"}

<br>

<br>

- Least square classification
- Perceptron
- Logistic regression

What is a common among all these models?

:::

:::



# Story so far

::: {.columns align=left}

::: {.column width="50%"}

![](images/003.svg)

:::

::: {.column width="50%"}

<br>

<br>

- Least square classification
- Perceptron
- Logistic regression

What is a common among all these models?

:::

:::



# Boundaries

::: {.columns align=center}

::: {.column width="50%"}

<br>

<br>

:::

::: {.column width="50%"}

:::

:::



# Boundaries

::: {.columns align=center}

::: {.column width="100%"}

![](images/002.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# Boundaries

::: {.columns align=center}

::: {.column width="100%"}

![](images/005.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# Boundaries

::: {.columns align=center}

::: {.column width="100%"}

![](images/004.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# "Best" Boundary?

::: {.columns align=center}

::: {.column width="100%"}

![](images/006.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# "Best" Boundary?

::: {.columns align=center}

::: {.column width="100%"}

![](images/007.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# "Best" Boundary?

::: {.columns align=center}

::: {.column width="100%"}

![](images/008.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::

# "Best" Boundary?

::: {.columns align=center}

::: {.column width="100%"}

![](images/009.svg){width="900"}

:::

::: {.column width="0%"}

:::

:::



# "Best" Boundary?

::: {.columns align=left}

::: {.column width="100%"}

<br>

<br>

::: incremental

- A decision boundary that is  "pointo-phobic" is a good one.

- Stay away from data-points of either class.
- The most pointo-phobic boundary is the best one.
- The "middle path".

:::

:::

::: {.column width="0%"}

:::

:::





