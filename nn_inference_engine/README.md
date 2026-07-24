# Neural Network Inference Engine

## Overview

Simple feedforward neural network (C++17).

The network consists of:

- Two hidden dense layers
- ReLU activation
- Softmax output
- Xavier (Glorot) weight initialization
- Binary model serialization
- Feedforward inference

The NN predicts one of four output classes from sixteen input features.

---
## C++ (justifications)

Production inference often values latency, predictable memory use, binary deployment, and integration with native systems.

This NN implements matrix-vector operations, ReLU, softmax, and binary model serialization without a heavyweight runtime.


## Concepts demonstrated

- Dense layers
- Matrix-vector multiplication
- ReLU
- Numerically stable softmax
- Binary model serialization
---

## Network Architecture

```text
Input (16)
      │
      ▼
Dense Layer (32)
      │
    ReLU
      │
      ▼
Dense Layer (16)
      │
    ReLU
      │
      ▼
Dense Layer (4)
      │
   Softmax
      │
      ▼
Class Probabilities
```

---

## Forward Propagation

Each neuron computes

```math
z = Wx + b
```

where:

- $z$: Output before activation.
- $W$: Weight matrix.
- $x$: Input vector.
- $b$: Bias vector.

Each neuron multiplies the input values by their corresponding weights, adds the bias, and produces an output.

---

## ReLU Activation

Hidden layers use the Rectified Linear Unit (ReLU):

```math
\operatorname{ReLU}(x)=\max(0,x)
```

where:

- $\operatorname{ReLU}$: Rectified Linear Unit activation.
- $x$: Input value.
- $\max(0,x)$: Returns the larger of 0 or $x$.

ReLU removes negative values while keeping positive values unchanged.

---

## Softmax

The output layer converts logits into probabilities.

```math
P_i=
\frac{e^{z_i}}
{\sum_{j=1}^{n}e^{z_j}}
```

where:

- $P_i$: Probability of class $i$.
- $e$: Euler's number (approximately 2.71828).
- $z_i$:Logit for class $i$.
- $\sum$: Sum over all output classes.
- $n$: Number of output classes.

The probabilities always satisfy

```math
\sum_i P_i = 1
```

meaning they add up to exactly 1.

---

## Xavier Initialization

Weights are initialized using Xavier initialization.

Xavier initialization = technique for setting the starting weights of a neural network

(Xavier initialization) used to prevent vanishing or exploding gradients and keep signal variance stable across deep neural network layers. 

It sets the variance of weights based on the number of input and output connections, ensuring effective training for activation functions like sigmoid or tanh

```math
\text{limit}
=
\sqrt{
\frac{6}
{\text{input size}+\text{output size}}
}
```

where:

- $\sqrt{\phantom{x}}$: Square root.
- $6$: Constant from Xavier initialization.
- input size: Number of input neurons.
- output size: Number of output neurons.

Weights are then sampled uniformly from

```math
[-\text{limit},\text{limit}]
```

This helps prevent activations from becoming too large or too small during forward propagation.

---

## Prediction

The predicted class is the output neuron with the highest probability.


```text
Class 0: 0.289404
Class 1: 0.345848
Class 2: 0.187449
Class 3: 0.177298
```

Since Class 1 has the highest probability,

```text
Predicted Class = 1
```


---

## Output

```text
Class probabilities:
class 0: 0.289404
class 1: 0.345848
class 2: 0.187449
class 3: 0.177298

predicted class: 1

restored model first probability: 0.289404
```

## Results

The neural network (successfully) performed a forward pass using a sample input containing **16 features** and produced a probability for each of the **4 output classes**.

```text
Class probabilities:
class 0: 0.289404
class 1: 0.345848
class 2: 0.187449
class 3: 0.177298

predicted class: 1

restored model first probability: 0.289404
```

### Model Flow

- The network predicts that the input belongs to **Class 1** because it has the highest probability (**34.58%**).
- Softmax converts the network's raw outputs (logits) into probabilities that sum to approximately **1.0**.
- The remaining classes receive lower probabilities, indicating that the network considers them less likely.

### Model Strengths

- Implements a complete **feedforward neural network** entirely in modern C++17.
- Uses **dense (fully connected) layers** for feature learning.
- Applies the **ReLU activation function** to introduce non-linearity.
- Uses **Softmax** to produce normalized class probabilities.
- Initializes weights using **Xavier (Glorot) initialization** for improved numerical stability.
- Supports **binary model serialization**, allowing trained parameters to be saved and restored.
- Performs **deterministic inference** by using a fixed random seed (`42`), producing reproducible results.

- simple implmentation of neural network architecture, forward propagation, activation functions, Softmax classification, and model serialization.