# Low-Level Machine Learning (C++)

### This repo is made up of 3 ML-Based projects (C++):

### **Note 1 (READ):** Descriptive Math-Heavy README's located within algorithm directories (open each directory to see)
### **Note 2:** A C++17-compatible compiler is required.

1. [**Linear Regression**](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/) **with** [**Gradient Descent**](https://en.wikipedia.org/wiki/Gradient_descent)
   - Dense numerical loops
   - Explicit memory layout
   - Fast batch training without Python interpreter overhead

2. [**Parallel K-Means Clustering**](https://users.ece.northwestern.edu/~wkliao/Kmeans/index.html)
   - CPU-bound distance calculations
   - Multi-threaded assignment step
   - Large in-memory datasets

3. [**Neural Network Inference Engine**](https://en.wikipedia.org/wiki/Inference_engine)
   - Low-latency inference
   - Contiguous parameter storage
   - Manual matrix-vector kernels
   - No heavyweight runtime dependency
