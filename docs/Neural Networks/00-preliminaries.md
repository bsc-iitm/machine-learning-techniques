# Preliminaries

## Problems

We will discuss two types of problems:

- Regression
- Multiclass classification

We won't be discussing binary classification explicitly. Most of the discussion about neural networks apply to both classes of problems. There are a handful of places where the network has to be specified differently for regression and classification. Watch out for these instances.



## Notation

Scalars will be represented using normal font, lower-case letters. Vectors will be represented using bold font, lower-case letters. Matrices will be represented using bold font, upper-case letters. When indexing elements of a vector or a matrix, normal font will be used, but the case will be inherited from the object that is being indexed.

- $a$: scalar

- $\boldsymbol{a}$: vector

- $\boldsymbol{A}$: matrix


$a_i$: $i^{\text{th}}$ component of the vector and $A_{ij}$: $j^{th}$ element in the $i^{\text{th}}$ row of the matrix. Indices are used minimally as most of the equations are vectorized. This is a convention that we will largely stick to. But we might have to override them in a few occasions. In such situations, the nature of the object should be inferred from the context.



## Data

The data-matrix is $\boldsymbol{X}$:



- size: $n \times m$
- $n$ data-points
- $m$ features



The data-matrix is common to both problems. Labels are represented differently in the case of regression and classification:



### Regression

The predicted labels for regression is $\boldsymbol{y}$, a vector of real numbers:



- size: $n$
- $n$ data-points
- single target corresponding to each point



### Multiclass Classification

The one-hot matrix of labels for a multiclass classification problem is $\boldsymbol{Y}$:



- size: $n \times k$
- $n$ data-points
- $k$ classes



Each row of the matrix is a one-hot vector.
