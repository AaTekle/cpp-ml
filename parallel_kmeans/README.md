# Parallel K-Means Clustering (C++17)

K-Means is an **unsupervised learning algorithm**. It discovers groups in data without requiring known class labels.

This project implements the **K-Means clustering algorithm**.

K-Means clustering algorithm:

* Generates a synthetic dataset containing four natural groups.
* Stores the dataset in contiguous row-major memory.
* Initializes four centroids.
* Assigns points to their nearest centroids.
* Recalculates each centroid as a cluster mean.
* Repeats until the centroids stop moving.
* Uses multiple CPU threads to parallelize point assignment.
* Prints the final coordinates of each discovered centroid.
---

## Dataset Size

The program creates:

* 50,000 points per cluster
* 4 clusters
* 8 dimensions per point


The total number of points is:

```math
N = 50{,}000 \times 4 = 200{,}000
```

where:

* $N$ : Total number of points.
* $50{,}000$ : Number of points generated for each cluster.
* $\times$ : Multiplication operator.
* $4$ : Number of clusters.
* $200{,}000$ : Total number of generated points.

Each point contains eight numerical features.

The complete dataset then stores:

```math
200{,}000 \times 8 = 1{,}600{,}000
```

double-precision values.

---

## Point-Cloud Representation

The dataset is stored in the following structure:

```cpp
struct PointCloud {
    std::size_t rows{};
    std::size_t dims{};
    std::vector<double> values;
};
```

* `rows` : Number of points.
* `dims` : Number of dimensions or features per point.
* `values` : Contiguous storage containing all coordinates.

The storage is **row-major**, meaning all dimensions of one point are stored next to each other.

A coordinate is retrieved using:

```math
\text{index}=iD+d
```

where:

* $\text{index}$ : Position inside the one-dimensional vector.
* $i$ : Point or row index.
* $D$ : Number of dimensions.
* $d$ : Dimension index.
* Multiplication joins the row offset with the dimension count.
* Addition moves to the selected coordinate within the row.

In the code:

```cpp
data.values[row * data.dims + d]
```

For example, if a point has eight dimensions and the program needs dimension three of row ten:

```math
10 \times 8 + 3 = 83
```

The coordinate is stored at vector index `83`.

---

## Synthetic Cluster Generation

Each generated coordinate is based on a known cluster center:

```math
\mu_{c,d}=5c+0.4d
```

where:

* $\mu$ : Greek letter “mu,” commonly used to represent a mean or center.
* $\mu_{c,d}$ : Center for cluster $c$ in dimension $d$.
* $c$ : Cluster index.
* $d$ : Dimension index.
* $5$ : Separation between neighboring clusters.
* $0.4$ : Increase between neighboring dimensions.
* $+$ : Addition operator.
* The comma in the subscript separates the cluster index from the dimension index.

The actual generated coordinate is:

```math
x_{i,d}=\mu_{c,d}+\epsilon
```

where:

* $x_{i,d}$ : Dimension $d$ of point $i$.
* $i$ : Point index.
* $d$ : Dimension index.
* $\mu_{c,d}$ : Expected center of cluster $c$ in dimension $d$.
* $\epsilon$ : Greek letter “epsilon,” representing random noise.
* $+$ : Adds noise to the ideal cluster center.

---

## Gaussian Noise

Noise is sampled from:

```math
\epsilon \sim \mathcal{N}(0,0.65^2)
```

where:

* $\epsilon$ : Random noise added to a coordinate.
* $\sim$ : “Is distributed according to.”
* $\mathcal{N}$ : Normal or Gaussian distribution.
* $0$ : Mean of the noise distribution.
* $0.65$ : Standard deviation.
* $0.65^2$ : Variance.
* The exponent $2$ means that `0.65` is multiplied by itself.

The variance is:

```math
0.65^2=0.4225
```

Because the mean is zero, the noise does not systematically push coordinates upward or downward.

Most values remain relatively close to their cluster center, while some points are farther away.

---

# K-Means Algorithm

K-Means alternates between two main steps:

1. Assign each point to its nearest centroid.
2. Recalculate each centroid using the assigned points.

These steps repeat until the centroids stop moving or the maximum number of iterations is reached.

---

## Step 1: Initialize Centroids

K-Means needs an initial location for each centroid.

K-Means chooses evenly spaced rows from the dataset:

```cpp
const std::size_t source_row =
    cluster * (data.rows - 1) /
    std::max<std::size_t>(1, k_ - 1);
```

This initialization is deterministic.

Because the generated clusters are stored in order, the selected points are spread across the dataset and are likely to come from different natural groups.

This helps the algorithm converge quickly in this example.

---

## Step 2: Calculate Distance

Each point is compared with every centroid using squared Euclidean distance.

```math
d^2(x,\mu_k)
=
\sum_{j=0}^{D-1}
(x_j-\mu_{k,j})^2
```

where:

* $d^2$ : Squared distance.
* $d$ : Distance function.
* The exponent $2$ indicates squared distance.
* $x$ : Current point.
* $\mu_k$ : Centroid of cluster $k$.
* $\mu$ : Greek letter “mu,” representing a centroid or mean.
* $k$ : Cluster index.
* $D$ : Number of dimensions.
* $\sum$ : Greek capital sigma, meaning “add all terms.”
* $j$ : Current dimension index.
* $j=0$ : Begin with the first dimension.
* $D-1$ : Final valid dimension index.
* $x_j$ : Coordinate of the point in dimension $j$.
* $\mu_{k,j}$ : Coordinate of centroid $k$ in dimension $j$.
* $-$ : Calculates the difference between coordinates.
* Parentheses group the difference.
* The exponent $2$ squares the difference.

The implementation is:

```cpp
const double delta = left[d] - right[d];
total += delta * delta;
```

The normal Euclidean distance is:

```math
d(x,\mu_k)
=
\sqrt{
\sum_{j=0}^{D-1}
(x_j-\mu_{k,j})^2
}
```

However, the assignment step does not need the square root.

The square-root function is increasing, so the smallest squared distance also has the smallest regular distance.

Removing the square root makes the comparison faster.

---

## Step 3: Assign Each Point

Each point is assigned to the centroid with the smallest distance:

```math
z_i
=
\underset{k}{\operatorname{argmin}}
\;
d^2(x_i,\mu_k)
```

where:

* $z_i$ : Cluster label assigned to point $i$.
* $i$ : Point index.
* $\operatorname{argmin}$ : Returns the index that produces the smallest value.
* $k$ : Candidate cluster index.
* $d^2$ : Squared Euclidean distance.
* $x_i$ : Point $i$.
* $\mu_k$ : Centroid of cluster $k$.

This equation does not return the distance itself.

It returns the **cluster index** whose centroid has the smallest distance.

For example, suppose a point has these squared distances:

```text
Cluster 0: 80.2
Cluster 1: 4.7
Cluster 2: 95.1
Cluster 3: 310.4
```

The smallest distance is `4.7`, so the point receives label `1`.

---

## Step 4: Recompute Centroids

After all points receive labels, each centroid becomes the mean of the points assigned to that cluster.

```math
\mu_k
=
\frac{1}{|C_k|}
\sum_{x_i\in C_k}
x_i
```

where:

* $\mu_k$ : Updated centroid for cluster $k$.
* $\mu$ : Greek letter “mu,” representing a mean.
* $k$ : Cluster index.
* $C_k$ : Set of points assigned to cluster $k$.
* $|C_k|$ : Number of points in cluster $k$.
* Vertical bars around a set represent its size.
* $\frac{1}{|C_k|}$ : Divides by the cluster’s point count.
* $\sum$ : Adds all assigned points.
* $x_i$ : Point $i$.
* $\in$ : “Belongs to.”
* $x_i\in C_k$ : Point $i$ belongs to cluster $k$.

For one coordinate, the calculation is:

```math
\mu_{k,d}
=
\frac{
\sum_{i:z_i=k}x_{i,d}
}{
n_k
}
```

where:

* $\mu_{k,d}$ : Dimension $d$ of centroid $k$.
* $x_{i,d}$ : Dimension $d$ of point $i$.
* $z_i$ : Cluster assigned to point $i$.
* $z_i=k$ : Only include points assigned to cluster $k$.
* $n_k$ : Number of points assigned to cluster $k$.
* The fraction divides the coordinate sum by the number of points.

---

## Step 5: Measure Centroid Shift

After recalculating the centroids, the program measures their total movement.

```math
S
=
\sqrt{
\sum_{k=0}^{K-1}
\sum_{d=0}^{D-1}
\left(
\mu_{k,d}^{\text{new}}
-
\mu_{k,d}^{\text{old}}
\right)^2
}
```

where:

* $S$ : Total centroid shift.
* $\sqrt{\phantom{x}}$ : Square-root operator.
* $\sum$ : Adds movement across all clusters and dimensions.
* $k$ : Cluster index.
* $K$ : Total number of clusters.
* $d$ : Dimension index.
* $D$ : Total number of dimensions.
* $\mu_{k,d}^{\text{new}}$ : New centroid coordinate.
* $\mu_{k,d}^{\text{old}}$ : Previous centroid coordinate.
* Superscripts `new` and `old` identify the updated and previous values.
* $-$ : Difference between the two centroid coordinates.
* The exponent $2$ makes every contribution nonnegative.

The shift describes how much all centroid coordinates moved during one iteration.

---

## Stopping Condition

The algorithm stops when:

```math
S<\tau
```

where:

* $S$ : Current centroid shift.
* $<$ : Less-than comparison.
* $\tau$ : Greek letter “tau,” representing the convergence tolerance.
* In this program, $\tau=10^{-6}$.

The configured tolerance is:

```cpp
1e-6
```

This is scientific notation for:

```math
1\times10^{-6}=0.000001
```

If the shift is smaller than `0.000001`, the centroids are considered stable.

The algorithm also stops if it reaches 50 iterations.

---

# Parallel Processing

The most expensive part of K-Means is assigning points to clusters.

Each point can be compared with the centroids independently, so the program divides the dataset into blocks and processes those blocks with multiple threads.

---

## Thread Count

The default thread count comes from:

```cpp
std::thread::hardware_concurrency()
```

This provides an estimate of the number of hardware execution threads supported by the CPU.

The program will make sure that at least one thread is used:

```cpp
std::max<std::size_t>(1, thread_count)
```

It also prevents creating more worker threads than data rows:

```cpp
std::min(thread_count_, data.rows)
```

---

## Block Size

The number of rows assigned to each thread is calculated as:

```math
B
=
\left\lceil
\frac{N}{T}
\right\rceil
```

where:

* $B$ : Block size.
* $N$ : Number of data points.
* $T$ : Number of active threads.
* $\lceil\phantom{x}\rceil$ : Ceiling operator, which rounds upward.
* The fraction divides the points among the threads.

The code uses integer arithmetic:

```cpp
(data.rows + actual_threads - 1) / actual_threads
```

This produces ceiling division without using floating-point calculations.

Each worker receives a range:

```text
[begin, end)
```

The opening bracket means `begin` is included.

The closing parenthesis means `end` is excluded.

---

## Why Assignment Is Thread-Safe

Each thread:

* Reads the same point data.
* Reads the same centroid data.
* Writes labels for a different range of rows.

No two threads write to the same label position.

then, the assignment step does not require a mutex.

All threads are joined before centroid recomputation begins:

```cpp
worker.join();
```

This ensures that every label has been assigned before the next algorithm step.

---

# K-Means Objective Function

K-Means attempts to minimize the within-cluster sum of squared distances:

```math
J
=
\sum_{i=1}^{N}
\left\|
x_i-\mu_{z_i}
\right\|_2^2
```

where:

* $J$ : K-Means objective function.
* $\sum$ : Adds the error for every point.
* $i$ : Point index.
* $N$ : Number of points.
* $x_i$ : Point $i$.
* $z_i$ : Cluster assigned to point $i$.
* $\mu_{z_i}$ : Centroid of the cluster assigned to point $i$.
* $|\cdot|_2$ : Euclidean or L2 norm.
* The subscript $2$ identifies Euclidean distance.
* The outer exponent $2$ squares the distance.

A smaller value of $J$ means points are generally closer to their assigned centroids.

The script does not print $J$, but the assignment and centroid-update steps are designed to reduce it.

---

# Computational Complexity

For each iteration, every point is compared with every centroid across every dimension.

The approximate assignment complexity is:

```math
O(NKD)
```

where:

* $O$ : Big-O notation describing how runtime grows.
* $N$ : Number of points.
* $K$ : Number of clusters.
* $D$ : Number of dimensions.
* Multiplication means all three factors affect the work required.

For this program:

```math
N=200{,}000
```

```math
K=4
```

```math
D=8
```

Distance-coordinate comparisons per iteration are approximately:

```math
200{,}000\times4\times8
=
6{,}400{,}000
```

With $T$ effective threads, the idealized assignment time becomes approximately:

```math
O\left(\frac{NKD}{T}\right)
```

Actual speedup may be smaller because of:

* Thread creation overhead.
* Memory bandwidth.
* CPU scheduling.
* Cache behavior.
* Serial centroid recomputation.

---

# Results

The program produces:

```text
Iteration   0 | centroid shift: 4.02235478
Iteration   1 | centroid shift: 0.00000000

Final centroids:
Cluster 0: 0.000 0.401 0.798 1.198 1.605 2.003 2.399 2.803
Cluster 1: 4.998 5.396 5.800 6.201 6.599 6.999 7.398 7.804
Cluster 2: 9.998 10.403 10.797 11.205 11.597 12.003 12.405 12.798
Cluster 3: 14.999 15.400 15.802 16.202 16.596 16.998 17.402 17.799
```

---

## Iteration 0

```text
centroid shift: 4.02235478
```

This means the centroids moved a combined Euclidean distance of approximately `4.02235478` after the first assignment and update step.

The original centroids were individual data points selected during initialization.

After the first update, each centroid became the mean of approximately 50,000 points.

The movement is expected because a randomly noisy sample point is unlikely to lie exactly at the true center of its cluster.

---

## Iteration 1

```text
centroid shift: 0.00000000
```

A shift of zero means:

* Point assignments did not change.
* Recomputed cluster means were identical to the previous centroids.
* The algorithm reached a stable solution.
* Additional iterations would produce the same centroids.

Because:

```math
0<10^{-6}
```

the stopping condition is satisfied.

The model converged after only two printed iterations.

---

## Expected Cluster Centers

The synthetic data was generated using:

```math
\mu_{c,d}=5c+0.4d
```

For Cluster 0, where $c=0$:

```math
\mu_{0,d}=0.4d
```

The expected coordinates are:

```text
0.0  0.4  0.8  1.2  1.6  2.0  2.4  2.8
```

The learned centroid is:

```text
0.000 0.401 0.798 1.198 1.605 2.003 2.399 2.803
```

These values are extremely close to the expected center.

where:

* $\mu$ : Greek letter “mu,” commonly used to represent a mean or center.
* $\mu_{c,d}$ : Expected center of cluster $c$ in dimension $d$.
* $c$ : Cluster index.
* $d$ : Dimension index.
* The comma in the subscript separates the cluster index from the dimension index.
* $5c$ : Determines the general location of the cluster.
* $5$ : Distance placed between neighboring cluster centers.
* $0.4d$ : Determines how the center changes across dimensions.
* $0.4$ : Amount added for each new dimension.

---

## Cluster 1

For $c=1$:

```math
\mu_{1,d}=5+0.4d
```
where:

* $\mu_{1,d}$ : Expected center of Cluster 1 in dimension $d$.
* $1$ : Identifies Cluster 1.
* $5$ : Base position of the cluster.
* $0.4d$ : Increase across dimensions.
Expected:

```text
5.0  5.4  5.8  6.2  6.6  7.0  7.4  7.8
```

Learned:

```text
4.998 5.396 5.800 6.201 6.599 6.999 7.398 7.804
```

The differences are only a few thousandths.

---

## Cluster 2

For $c=2$:

```math
\mu_{2,d}=10+0.4d
```
where:

* $\mu_{2,d}$ : Expected center of Cluster 2 in dimension $d$.
* $2$ : Identifies Cluster 2.
* $10$ : Base location of the cluster.
* $0.4d$ : Dimension-dependent increase.

Expected:

```text
10.0 10.4 10.8 11.2 11.6 12.0 12.4 12.8
```

Learned:

```text
9.998 10.403 10.797 11.205 11.597 12.003 12.405 12.798
```

Again, the learned center closely matches the true center.

---

## Cluster 3

For $c=3$:

```math
\mu_{3,d}=15+0.4d
```
where:

* $\mu_{3,d}$ : Expected center of Cluster 3 in dimension $d$.
* $3$ : Identifies Cluster 3.
* $15$ : Base location of the cluster.
* $0.4d$ : Increase applied across dimensions.

Expected:

```text
15.0 15.4 15.8 16.2 16.6 17.0 17.4 17.8
```

Learned:

```text
14.999 15.400 15.802 16.202 16.596 16.998 17.402 17.799
```

The centroid is nearly identical to the synthetic cluster center.

---

# Results - Analysis

The results closely match the known centers because:

* Each cluster contains 50,000 points.
* Gaussian noise has a mean of zero.
* Positive and negative noise largely cancel when averaged.
* The clusters are separated by approximately 5.0 units.
* The noise standard deviation is only 0.65.
* Initial centroids are spread across the ordered dataset.
* The four natural groups are well separated.

For a cluster coordinate:

```math
\bar{x}
=
\mu+
\frac{1}{n}
\sum_{i=1}^{n}\epsilon_i
```

where:

* $\bar{x}$: Average coordinate.
* The bar above $x$ means mean.
* $\mu$: True cluster center.
* $n$: Number of points.
* $\epsilon_i$: Noise added to point $i$.
* $\sum$: Adds all noise values.

As $n$ becomes large:

```math
\frac{1}{n}
\sum_{i=1}^{n}\epsilon_i
\approx0
```

Therefore:

```math
\bar{x}\approx\mu
```

With 50,000 points per cluster, the sample mean becomes very close to the true center.

---

## Model Limitations

* The generated data is simpler than most real-world datasets.
* K-Means is sensitive to feature scale.
* Empty clusters are skipped rather than reinitialized.
