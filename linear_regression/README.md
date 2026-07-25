# Multivariate Linear Regression (C++17)

multivariate linear (C++17) using **batch gradient descent**.

**Goal:** Multivariate Linear Regression using low-level numerical computation (without relying on external machine learning libraries, for learning purposes).

---

## Linear Regression

Linear regression is a supervised machine learning algorithm used to predict a continuous numerical value.

e.g. include:

* Predicting house prices
* Estimating revenue
* Forecasting energy consumption
* Predicting customer lifetime value
* Estimating delivery times
* Modeling relationships between numerical variables

The algorithm assumes that the target variable can be approximated as a linear combination of the input features.

For one feature, the model is:

```math
\hat{y} = b + wx
```

For multiple features, the model becomes:

```math
\hat{y} = b + w_0x_0 + w_1x_1 + \cdots + w_{p-1}x_{p-1}
```

where:

* $\hat{y}$ is the predicted value
* $b$ is the bias or intercept
* $x_j$ is feature $j$
* $w_j$ is the learned coefficient for feature $j$
* $p$ is the number of features

Each coefficient represents how much the prediction changes when its corresponding feature increases by one unit while all other features remain constant.

For example, if:

```math
w_2 = 2.25
```

then increasing $x_2$ by one unit increases the prediction by approximately $2.25$, assuming the other features do not change.

---

## C++? (justifications)

Training a linear regression model consists of repeated numerical operations:

* Dense vector and matrix multiplication
* Dot products
* Gradient computation
* Parameter updates
* Repeated iteration over large datasets
* Sequential access to contiguous numerical arrays

These operations are CPU-intensive/dependent and benefit from C++ because C++ provides:

* Predictable contiguous memory layout using `std::vector`
* Minimal runtime overhead
* Compiler auto-vectorization with optimizations such as `-O3`
* Fine-grained control over memory allocation
* Direct pointer access to matrix rows
* Strong type safety
* Efficient value semantics and move semantics

C++ allows the mathematical kernels to execute with very little abstraction compared with interpreted or dynamically dispatched code.

C++ (context of ML) is useful when:

* Inference latency matters
* predictable memory usage 
* parallelized numerical computation
---

## Concepts Applied

* Supervised Machine Learning
* Multivariate Linear Regression
* Batch Gradient Descent
* Mean Squared Error
* Root Mean Squared Error
* Mean Absolute Error
* Coefficient of Determination
* Dense numerical computation
* Row-major feature matrices
* Synthetic dataset generation
* Parameter optimization
* Train and test evaluation
* Contiguous memory
* Pointer-based row access
* Exception handling
* C++17

---

## Dataset

The feature matrix contains:

* 20,000 training observations
* 5,000 test observations
* 5 features per observation

Conceptually, the feature matrix is two-dimensional:

```math
X =
\begin{bmatrix}
x_{00} & x_{01} & x_{02} & x_{03} & x_{04} \\
x_{10} & x_{11} & x_{12} & x_{13} & x_{14} \\
\vdots & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}
```
where:

* $X$ is the complete feature matrix.
* Each row represents one observation, sample, or data record.
* Each column represents one input feature.
* $x_{ij}$ represents the value at row $i$ and column $j$.
* $i$ identifies the observation.
* $j$ identifies the feature.
* In memory, the matrix is stored as one contiguous `std::vector<double>`:

```text
row 0 features, row 1 features, row 2 features, ...
```

The value at row `i` and column `j` is accessed with:

```cpp
x[i * cols + j]
```

This is called **row-major storage**.

A pointer to the beginning of a row is obtained with:

```cpp
const double* row = &data.x[i * data.cols];
```

the current row can then be accessed as:

```cpp
row[0]
row[1]
row[2]
```

this avoids constructing or copying a separate vector for every observation.

---

## Synthetic Data Generation
Project Focus: algorithmic implementation

Using synthetic data as a placeholder:

The synthetic dataset is generated from the equation:

```math
y = 1.5 + 0.75x_0 + 1.50x_1 + 2.25x_2 + 3.00x_3 + 3.75x_4 + \epsilon
```

The true parameters are:

```math
b = 1.5
```

```math
w =
\begin{bmatrix}
0.75 \\
1.50 \\
2.25 \\
3.00 \\
3.75
\end{bmatrix}
```

Each feature is sampled from a standard normal distribution:

```math
x_j \sim \mathcal{N}(0, 1)
```
- $x_j$ is the value of feature $j$.
- $j$ is the feature (column) index.
- $\sim$ means **"is distributed as"** or **"is sampled from."**
- $\mathcal{N}$ = a **Normal (Gaussian) distribution**.
- The first parameter, $0$, is the **mean ($\mu$)** of the distribution.
  - A mean of 0 means feature values are centered around zero.
- The second parameter, $1$, is the **variance ($\sigma^2$)** of the distribution.
  - A variance of 1 corresponds to a standard deviation of:
    ```math
    \sigma = \sqrt{1} = 1
    ```


Random Gaussian noise is added to each target:

```math
\epsilon \sim \mathcal{N}(0, 0.25^2)
```
where:

- $\epsilon$ — The **random Gaussian noise** (random error) added to each target value.
- $\sim$ — Means **"is distributed as"** or **"follows the probability distribution."**
- $\mathcal{N}$ — The **Normal (Gaussian) distribution**, also called the **bell-curve distribution**.
- $(\ )$ — Parentheses containing the parameters that define the Normal distribution.
- $0$ — The **mean ($\mu$)** of the distribution, indicating that the average noise is zero. On average, the noise neither increases nor decreases the target values.
- $,$ — Separates the distribution's parameters (mean and variance).
- $0.25$ — The **standard deviation ($\sigma$)** of the distribution, which determines the typical size of the random noise.
- $2$ (the exponent) — Squares the standard deviation to produce the variance.
- $0.25^2$ — The **variance ($\sigma^2$)** of the distribution.

then:

```math
\operatorname{Var}(\epsilon) = 0.25^2 = 0.0625
```

noise makes the problem more realistic. Without noise, the model could (theoretically) recover the relationship with almost zero prediction error.

with noise, some part of the target is random and cannot be predicted from the features.

---

## Model Prediction

For observation $i$, the prediction is:

```math
\hat{y}_i = b + \sum_{j=0}^{p-1} w_jx_{ij}
```
- $\hat{y}_i$ is the **predicted target value** for observation $i$.
- $\hat{\phantom{y}}$ (the "hat") shows that the value is **predicted** rather than observed.
- $i$ is the observation (row) index.
- $b$ is the **bias (intercept)**. It represents the baseline prediction before considering any feature values.
- $+$ shows that the weighted contributions of each feature are added to the bias.
- $\sum$ (capital Sigma) is the **summation operator**, meaning "add together."
- $j$ is the feature (column) index.
- $j = 0$ shows that the summation begins with the **first feature**. Since C++ uses **zero-based indexing**, the first feature has index `0`.
- $p$ is the total number of input features.
- $p - 1$ is the index of the **last feature**. Because indexing begins at `0`, the final feature is located at index `p - 1`.
- $w_j$ is the learned **weight (coefficient)** associated with feature $j$.
- $x_{ij}$ is the value of feature $j$ for observation $i$.
- $w_jx_{ij}$ is the **weighted contribution** of feature $j$ to the prediction.


For this project:

```math
\hat{y}_i =
b +
w_0x_{i0} +
w_1x_{i1} +
w_2x_{i2} +
w_3x_{i3} +
w_4x_{i4}
```

implementation performs this calculation using a loop:

```cpp
double prediction = bias_;

for (std::size_t j = 0; j < data.cols; ++j) {
    prediction += row[j] * weights_[j];
}
```

this loop computes a dot product between:

* the current feature row
* the model's weight vector

---

## Residual Error

For each observation, the residual is:

```math
e_i = \hat{y}_i - y_i
```

where:

* $\hat{y}_i$ is the prediction
* $y_i$ is the actual target
* $e_i$ is the prediction error

a positive residual means the model predicted too high.

a negative residual means the model predicted too low.

the implementation calculates:

```cpp
const double error = prediction - data.y[i];
```

---

## Loss Function

The model uses Mean Squared Error as its training objective.

```math
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
```

where:

* $n$ is the number of observations
* $\hat{y}_i$ is the prediction
* $y_i$ is the actual value

the error is squared for three main reasons:

1. positive and negative residuals cannot cancel each other.
2. parge errors are penalized more heavily.
3. function is differentiable, which makes gradient-based optimization possible.

implementation accumulates squared error with:

```cpp
squared_error_sum += error * error;
```

then we compute the mean:

```cpp
const double mse =
    squared_error_sum / static_cast<double>(data.rows);
```

---

## Gradient Descent

goal of training: find the weights and bias that minimize Mean Squared Error.

model parameters are:

```math
\theta =
\begin{bmatrix}
b \\
w_0 \\
w_1 \\
\vdots \\
w_{p-1}
\end{bmatrix}
```

Gradient descent repeatedly:

1. makes predictions
2. calculates errors
3. computes the gradient of the loss
4. moves the parameters in the direction that reduces the loss

the general update rule is:

```math
\theta := \theta - \alpha \nabla J(\theta)
```

where:

* $\theta$ represents the parameters
* $\alpha$ is the learning rate
* $J(\theta)$ is the loss function
* $\nabla J(\theta)$ is the gradient

---

## Gradient of Mean Squared Error

The loss function is:

```math
J(w,b)
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
```

The prediction is:

```math
\hat{y}_i = b + \sum_{j=0}^{p-1}w_jx_{ij}
```
- $J(w,b)$ is the **loss (cost) function**. It measures the overall prediction error of the current model as a function of the weights ($w$) and bias ($b$).
- $w$ represents the vector containing all model weights (coefficients).
- $b$ is the model's bias (intercept).
- $\frac{1}{n}$ computes the average error by dividing the total error by the number of observations.
- $n$ is the total number of training observations (samples).
- $\sum$ (capital Sigma) is the **summation operator**, (meaning add together).
- $i$ is the observation index (row index).
- $i = 1$ shows that the summation starts at the first observation.
- $n$ above the Sigma shows that the summation continues until the final observation.
- $\hat{y}_i$ is the model's predicted value for observation $i$.
- $y_i$ is the actual (true) target value for observation $i$.
- $(\hat{y}_i-y_i)$ is called the **residual** or **prediction error**.
- $(\hat{y}_i-y_i)^2$ squares the prediction error so that:
  - Positive and negative errors cannot cancel each other.
  - Larger errors receive a much larger penalty.
  - The loss function remains differentiable, allowing gradient descent to optimize it.

### Weight Gradient

the partial derivative with respect to weight $w_j$ is:

```math
\frac{\partial J}{\partial w_j}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)x_{ij}
```
  - Weight Gradient measures **how much the loss changes** when weight $w_j$ changes by a small amount while all other weights remain fixed.
  - The gradient tells the optimization algorithm which direction to adjust the weight in order to reduce the loss.

- $\partial$ (the curly "d") denotes a **partial derivative**.
  - A partial derivative is used because the loss function depends on **multiple variables** (the bias and every weight), and we differentiate with respect to only one of them.

- $J$ is the **loss (cost) function**, specifically the Mean Squared Error (MSE).

- $w_j$ is the weight (coefficient) associated with feature $j$.

- $=$ means that the gradient is equal to the expression on the right-hand side.

- $\frac{2}{n}$ is a constant scaling factor.
  - $2$ comes from differentiating the squared error term:
    ```math
    (error)^2
    ```
  - $n$ is the total number of training observations (samples).
  - Dividing by $n$ computes the **average gradient** across the entire training dataset.

- $\sum$ (capital Sigma) is the **summation operator**, meaning "add together."

- $i$ is the observation (row) index.

- $i = 1$ shows that the summation begins with the first observation.

- $n$ above the Sigma shows that the summation continues until the final observation.

- $\hat{y}_i$ is the predicted target value for observation $i$.

- $\hat{\phantom{y}}$ (the "hat") shows a **predicted** value.

- $y_i$ is the actual (true) target value for observation $i$.

- $(\hat{y}_i-y_i)$ is the **prediction error (residual)** for observation $i$.
  - A positive value means the prediction is too large.
  - A negative value means the prediction is too small.

- $x_{ij}$ is the value of feature $j$ for observation $i$.

- $(\hat{y}_i-y_i)x_{ij}$ multiplies the prediction error by the corresponding feature value.
  - This determines how much feature $j$ contributed to the prediction error for observation $i$.
  - Larger feature values generally produce larger gradient contributions.


the implementation first accumulates:

```cpp
weight_gradient[j] += error * row[j];
```

after all observations have been processed, it applies:

```cpp
const double scale = 2.0 / static_cast<double>(data.rows);
```

the weight update is then:

```cpp
weights_[j] -=
    learning_rate_ *
    scale *
    weight_gradient[j];
```

This corresponds to:

```math
w_j :=
w_j -
\alpha
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)x_{ij}
```

### Bias Gradient

the partial derivative with respect to the bias is:

```math
\frac{\partial J}{\partial b}
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
```
where:

- $\frac{\partial J}{\partial b}$ — The **partial derivative** (gradient) of the loss function with respect to the bias parameter.
- $\partial$ — The **partial derivative symbol**, indicating that the derivative is taken with respect to only one variable while all other variables remain constant.
- $J$ — The **loss (cost) function**, representing the total prediction error across the training dataset.
- $b$ — The **bias (intercept)** of the linear regression model.
- $=$ — shows that the expression on the left is equal to the expression on the right.
- $\frac{2}{n}$ — The scaling factor applied to the average gradient.
- $2$ — Comes from differentiating the squared error term in the Mean Squared Error loss function.
- $n$ — The total number of observations (samples) in the dataset.
- $\sum$ — The uppercase Greek letter **Sigma**, representing the **summation operator**, meaning "add together."
- $i$ — The observation (row) index.
- $i=1$ — shows that the summation begins with the first observation.
- $n$ (above the Sigma) — shows that the summation continues through the final observation.
- $\hat{y}_i$ — The predicted target value for observation $i$.
- $\hat{\phantom{y}}$ — The **hat symbol**, indicating that the value is predicted or estimated rather than directly observed.
- $y_i$ — The actual (true) target value for observation $i$.
- $-$ — The subtraction operator, used to compute the prediction error.
- $(\hat{y}_i-y_i)$ — The prediction error (residual) for observation $i$.


the implementation accumulates
bias_gradient += error;

Then updates the bias:

```cpp
bias_ -=
    learning_rate_ *
    scale *
    bias_gradient;
```

---

## Batch Gradient Descent

the implementation uses the entire training dataset to calculate one gradient update.

for each epoch:

1. every training observation is processed.
2. all weight gradients are accumulated.
3. the bias gradient is accumulated.
4. the parameters are updated once.


### Batch Gradient Descent

* Uses the complete dataset for each update
* Produces stable gradients
* Is easy to understand

### Stochastic Gradient Descent

* Uses one observation per update
* Makes frequent, noisy updates
* Can scale well to large datasets

### Mini-Batch Gradient Descent

* Uses a subset of observations per update
* Balances stability and computational efficiency

---

## Learning Rate

The learning rate controls the size of each parameter update.

This project uses, Learning rate = 0.03


A learning rate that is too small may cause:

* Very slow convergence
* Excessive training time
* Minimal progress per epoch

A learning rate that is too large may cause:

* Overshooting the minimum
* Oscillating loss
* Divergence
* Numerical instability

the selected learning rate converged stably for the generated dataset.

---

## Epochs

One epoch is one complete pass through the training dataset.

This project runs 2,500 epochs

Each epoch processes all 20,000 training observations.

The model had effectively converged by approximately epoch 250. The remaining epochs did not materially improve the reported MSE.

this means the current implementation performs unnecessary work after convergence. 

---

## Results

### Training Loss

```text
Epoch    0 | MSE: 33.003891
Epoch  250 | MSE: 0.063108
Epoch  500 | MSE: 0.063108
Epoch  750 | MSE: 0.063108
Epoch 1000 | MSE: 0.063108
Epoch 1250 | MSE: 0.063108
Epoch 1500 | MSE: 0.063108
Epoch 1750 | MSE: 0.063108
Epoch 2000 | MSE: 0.063108
Epoch 2250 | MSE: 0.063108
Epoch 2499 | MSE: 0.063108
```

The loss decreased from:

```math
33.003891
```

to:

```math
0.063108
```

The percentage reduction is:

```math
\frac{33.003891 - 0.063108}{33.003891}
\times 100
\approx 99.81\%
```

The model then reduced its training loss by approximately **99.81%**.

The loss stopped changing at the displayed precision after epoch 250, indicating convergence.

---

## Learned Parameters

| Parameter | Expected |    Learned | Absolute Error |
| --------- | -------: | ---------: | -------------: |
| Bias      |   1.5000 | **1.5003** |         0.0003 |
| $w_0$     |   0.7500 | **0.7503** |         0.0003 |
| $w_1$     |   1.5000 | **1.4962** |         0.0038 |
| $w_2$     |   2.2500 | **2.2471** |         0.0029 |
| $w_3$     |   3.0000 | **2.9994** |         0.0006 |
| $w_4$     |   3.7500 | **3.7486** |         0.0014 |

The learned coefficients are extremely close to the true coefficients used to generate the data.

The largest absolute coefficient error is:

```math
|1.4962 - 1.5000| = 0.0038
```

This is a very small error and shows that gradient descent recovered the underlying linear relationship accurately.

---

## Evaluation Metrics - Regression

Regression models are evaluated using metrics that measure numerical prediction error.

---

### Mean Squared Error

Mean Squared Error is:

```math
\operatorname{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```
- $\operatorname{MSE}$ — **Mean Squared Error**, the average of the squared prediction errors.
- $=$ — shows that the expression on the left is equal to the expression on the right.
- $\frac{1}{n}$ — Divides the total squared error by the number of observations, producing the **average** squared error.
- $1$ — The numerator of the averaging fraction.
- $n$ — The total number of observations (samples) in the dataset.
- $\sum$ — The uppercase Greek letter **Sigma**, representing the **summation operator**, meaning "add together."
- $i$ — The observation (row) index.
- $i = 1$ — shows that the summation begins with the first observation.
- $n$ (above the Sigma) — shows that the summation continues until the final observation.
- $y_i$ — The actual (true) target value for observation $i$.
- $\hat{y}_i$ — The predicted target value for observation $i$.
- $\hat{\phantom{y}}$ (the "hat") — shows that the value is **predicted** rather than observed.
- $(y_i-\hat{y}_i)$ — The prediction error (residual) for observation $i$.
- $-$ — Computes the difference between the actual value and the predicted value.
- $(\ )$ — Parentheses ensure that the entire prediction error is calculated before it is squared.
- $2$ (the exponent) — Squares the prediction error, preventing positive and negative errors from canceling each other and penalizing larger errors more heavily.
- $(y_i-\hat{y}_i)^2$ — The squared prediction error for observation $i$.
Results:

```text
Training MSE = 0.0631
Test MSE     = 0.0644
```

Interpretation:

* Lower values are better.
* Zero indicates perfect predictions.
* Large errors receive a stronger penalty because residuals are squared.
* The metric is expressed in squared target units.

The test MSE is only slightly larger than the training MSE:

```math
0.0644 - 0.0631 = 0.0013
```

The relative increase is approximately:

```math
\frac{0.0644 - 0.0631}{0.0631}
\times 100
\approx 2.06\%
```

A gap of approximately 2% is small and indicates strong generalization.

---

### Root Mean Squared Error

Root Mean Squared Error is the square root of MSE:

```math
\operatorname{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
```
- $\operatorname{RMSE}$ **Root Mean Squared Error**, a metric that measures the average prediction error in the same units as the target variable.
- $=$ — Indicates that the expression on the left is equal to the expression on the right.
- $\sqrt{\phantom{x}}$ — The **square root** operator. It converts the Mean Squared Error back into the original units of the target variable.
- $\frac{1}{n}$ — Divides the total squared error by the number of observations, producing the **average** squared error.
- $1$ — The numerator of the averaging fraction.
- $n$ — The total number of observations (samples) in the dataset.
- $\sum$ — The uppercase Greek letter **Sigma**, representing the **summation operator**, meaning "add together."
- $i$ — The observation (row) index.
- $i = 1$ — shows that the summation begins with the first observation.
- $n$ (above the Sigma) — shows that the summation continues until the final observation.
- $y_i$ — The actual (true) target value for observation $i$.
- $\hat{y}_i$ — The predicted target value for observation $i$.
- $\hat{\phantom{y}}$ (the "hat") — Indicates that the value is **predicted** rather than observed.
- $(y_i-\hat{y}_i)$ — The prediction error (residual) for observation $i$.
- $-$ — Computes the difference between the actual value and the predicted value.
- $(\ )$ — Parentheses ensure that the entire prediction error is calculated before it is squared.
- $2$ (the exponent) — Squares the prediction error, preventing positive and negative errors from canceling each other and penalizing larger errors more heavily.
- $(y_i-\hat{y}_i)^2$ — The squared prediction error for observation $i$.

Using the test MSE:

```math
\operatorname{RMSE}
=
\sqrt{0.0644}
\approx 0.2538
```

and:

```text
Test RMSE ≈ 0.2538
```

Interpretations:

* RMSE is expressed in the same units as the target.
* A test RMSE of approximately 0.254 means predictions are typically about 0.254 target units away from the actual values, under the squared-error interpretation.
* This is almost identical to the noise standard deviation of 0.25.

that is expected because the irreducible random noise added to the targets has:

```math
\sigma = 0.25
```
---

### Coefficient of Determination

The coefficient of determination is:

```math
R^2
=
1 -
\frac{
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i-\bar{y})^2
}
```

where:

```math
\bar{y}
=
\frac{1}{n}
\sum_{i=1}^{n}y_i
```

- $R^2$ is the **Coefficient of Determination (R-squared)**, which measures how well the regression model explains the variation in the target variable.
- $1$ represents a perfect fit before accounting for the model's remaining prediction error.
- $-$ subtracts the proportion of unexplained variation from 1.
- $\frac{\text{Numerator}}{\text{Denominator}}$ is a ratio comparing the model's prediction error to the total variation present in the data.
- $\sum$ (capital Sigma) is the **summation operator**, meaning "add together."
- $i$ is the observation (row) index.
- $i=1$ means that the summation begins with the first observation.
- $n$ above the Sigma means that the summation continues until the final observation.
- $n$ is the total number of observations (samples).
- $y_i$ is the actual (true) target value for observation $i$.
- $\hat{y}_i$ is the model's predicted target value for observation $i$.
- $\hat{\phantom{y}}$ (the "hat") means a **predicted** value.
- $\bar{y}$ ("y-bar") is the mean (average) of all target values.
- $(y_i-\hat{y}_i)$ is the **prediction error (residual)** for observation $i$.
- $(y_i-\bar{y})$ is how far observation $i$ is from the average target value.
- $(y_i-\hat{y}_i)^2$ squares the prediction error so that larger errors receive a larger penalty and positive and negative errors cannot cancel each other.
- $(y_i-\bar{y})^2$ squares each observation's deviation from the mean, measuring its contribution to the total variability in the dataset.
- $\bar{y}$ ("y-bar") is the mean (average) target value.
- $\frac{1}{n}$ divides by the total number of observations.
- $\sum$ = add together.
- $i$ is the observation index.
- $i=1$ shows the first observation.
- $n$ shows the final observation.
- $y_i$ is the actual target value for observation $i$.



Interpretation:

* $R^2 = 1$ means perfect predictions.
* $R^2 = 0$ means the model performs no better than predicting the target mean.
* $R^2 < 0$ means the model performs worse than predicting the mean.

The current program does not calculate $R^2$.

Because the features are independently sampled from $\mathcal{N}(0,1)$, the approximate signal variance is:

```math
0.75^2 + 1.50^2 + 2.25^2 + 3.00^2 + 3.75^2
```

```math
= 0.5625 + 2.25 + 5.0625 + 9 + 14.0625
```

```math
= 30.9375
```

Including noise variance:

```math
\operatorname{Var}(y)
\approx 30.9375 + 0.0625
= 31
```

An approximate expected $R^2$ is then:

```math
R^2
\approx
1 - \frac{0.0644}{31}
\approx 0.9979
```

this is an approximation based on the known synthetic data-generating process. The exact $R^2$ should be calculated from the generated test targets.

---

## Noise Floor

the baseline level of random background error, gradient variance, or data artifacts that a model cannot automatically surpass

The synthetic targets include Gaussian noise with standard deviation:

```math
\sigma = 0.25
```

The variance is:

```math
\sigma^2 = 0.0625
```

Because the noise is random and independent of the features, a linear regression model cannot predict it.

The theoretical minimum expected MSE is then closer to:

```math
0.0625
```

The achieved test MSE is:

```math
0.0644
```

The difference is:

```math
0.0644 - 0.0625 = 0.0019
```

The test MSE is only approximately 3.04% above the theoretical noise variance:

```math
\frac{0.0644 - 0.0625}{0.0625}
\times 100
\approx 3.04\%
```

This shows that the model learned almost all predictable structure in the dataset.

A materially lower test MSE should not be expected unless:

* The random noise level is reduced
* The test sample happens to contain unusually small noise
* The model begins fitting random variation rather than true structure

---

## Generalization and Overfitting

Generalization describes how well a model performs on unseen data.

Training result:

```text
Training MSE = 0.0631
```

Test result:

```text
Test MSE = 0.0644
```

Because the values are close, the model generalizes well.

There is no meaningful evidence of overfitting.

A model would show stronger evidence of overfitting if:

* Training error continued to decrease
* Test error increased significantly
* Learned parameters became unstable
* Performance depended heavily on the training sample

That pattern is not present here.

---

## Underfitting

Underfitting usually occurs when a model is too simple or insufficiently trained to capture the underlying numerical relationship.

Evidence of underfitting would include:

* High training MSE
* High test MSE
* Coefficients far from the true values
* Loss that remains large after training

The model is not underfitting because:

* Training MSE is near the noise floor
* Test MSE is near the noise floor
* Learned parameters closely match the true parameters
* Training converged successfully

---

## Performance

| Metric                     |                Value |
| -------------------------- | -------------------: |
| Training Samples           |               20,000 |
| Test Samples               |                5,000 |
| Features                   |                    5 |
| Epochs                     |                2,500 |
| Learning Rate              |                 0.03 |
| Initial MSE                |              33.0039 |
| Final Training MSE         |               0.0631 |
| Test MSE                   |               0.0644 |
| Test RMSE                  | Approximately 0.2538 |
| Noise Standard Deviation   |                 0.25 |
| Noise Variance             |               0.0625 |
| Approximate Expected $R^2$ |               0.9979 |
| Loss Reduction             | Approximately 99.81% |

---

## Computational Complexity

Let:

* $n$ be the number of observations
* $p$ be the number of features
* $E$ be the number of epochs

For every observation, the algorithm:

* Computes a prediction across $p$ features
* Accumulates a gradient across $p$ features

The approximate training complexity is:

```math
O(E \cdot n \cdot p)
```
- $O(\cdot)$ denotes **Big-O notation**, which describes how the algorithm's running time grows as the input size increases. It focuses on the dominant growth rate rather than the exact execution time.
- $E$ is the total number of training **epochs**.
- $n$ is the total number of training observations (samples).
- $p$ is the total number of input features.
- $\cdot$ denotes **multiplication**.
For this project:

```math
E = 2500
```

```math
n = 20000
```

```math
p = 5
```

The model then performs on the order of:

```math
2500 \times 20000 \times 5
=
250,000,000
```

feature-level operations for prediction alone, with additional operations for gradient accumulation and updates.

The memory complexity is approximately:

```math
O(n \cdot p + n + p)
```

This includes:

* Feature matrix
* Target vector
* Model weights
* Gradient vector
