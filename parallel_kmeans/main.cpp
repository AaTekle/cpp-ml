#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

struct PointCloud {
    std::size_t rows{};
    std::size_t dims{};
    std::vector<double> values; // Stores points in row-major order
};

// Generates synthetic clustered data with Gaussian noise
PointCloud make_clustered_data(
    std::size_t points_per_cluster,
    std::size_t dimensions,
    std::size_t cluster_count,
    unsigned int seed = 42
) {
    if (points_per_cluster == 0 || dimensions == 0 || cluster_count == 0) {
        throw std::invalid_argument("all sizes must be positive");
    }

    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, 0.65);

    PointCloud cloud;
    cloud.rows = points_per_cluster * cluster_count;
    cloud.dims = dimensions;
    cloud.values.resize(cloud.rows * dimensions);

    for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
        for (std::size_t i = 0; i < points_per_cluster; ++i) {
            const std::size_t row = cluster * points_per_cluster + i;

            for (std::size_t d = 0; d < dimensions; ++d) {
                const double center =
                    static_cast<double>(cluster) * 5.0 +
                    static_cast<double>(d) * 0.4;

                cloud.values[row * dimensions + d] =
                    center + noise(generator);
            }
        }
    }

    return cloud;
}

class ParallelKMeans {
public:
    // Configures the K-Means model and its stopping conditions
    ParallelKMeans(
        std::size_t cluster_count,
        std::size_t max_iterations = 100,
        double tolerance = 1e-5,
        std::size_t thread_count = std::thread::hardware_concurrency()
    )
        : k_(cluster_count),
          max_iterations_(max_iterations),
          tolerance_(tolerance),
          thread_count_(std::max<std::size_t>(1, thread_count)) {
        if (k_ == 0) {
            throw std::invalid_argument("cluster_count must be positive");
        }
    }

    // Runs K-Means until convergence or the iteration limit is reached
    void fit(const PointCloud& data) {
        validate(data);

        centroids_.assign(k_ * data.dims, 0.0);
        labels_.assign(data.rows, 0);

        initialize_centroids(data);

        for (std::size_t iteration = 0;
             iteration < max_iterations_;
             ++iteration) {
            assign_clusters_parallel(data);

            const double shift = recompute_centroids(data);

            std::cout << "Iteration " << std::setw(3) << iteration
                      << " | centroid shift: "
                      << std::fixed << std::setprecision(8)
                      << shift << '\n';

            if (shift < tolerance_) {
                break;
            }
        }
    }

    // Returns the final centroid coordinates
    const std::vector<double>& centroids() const noexcept {
        return centroids_;
    }

    // Returns the cluster label assigned to each point
    const std::vector<std::size_t>& labels() const noexcept {
        return labels_;
    }

private:
    // Verifies that the point cloud has valid dimensions and storage
    void validate(const PointCloud& data) const {
        if (data.rows == 0 || data.dims == 0 ||
            data.values.size() != data.rows * data.dims) {
            throw std::invalid_argument("invalid point cloud");
        }

        if (k_ > data.rows) {
            throw std::invalid_argument("more clusters than points");
        }
    }

    // Selects initial centroids from evenly spaced rows
    void initialize_centroids(const PointCloud& data) {
        for (std::size_t cluster = 0; cluster < k_; ++cluster) {
            const std::size_t source_row =
                cluster * (data.rows - 1) /
                std::max<std::size_t>(1, k_ - 1);

            std::copy_n(
                &data.values[source_row * data.dims],
                data.dims,
                &centroids_[cluster * data.dims]
            );
        }
    }

    // Computes squared Euclidean distance between two points
    static double squared_distance(
        const double* left,
        const double* right,
        std::size_t dims
    ) {
        double total = 0.0;

        for (std::size_t d = 0; d < dims; ++d) {
            const double delta = left[d] - right[d];
            total += delta * delta;
        }

        return total;
    }

    // Assigns a range of points to their nearest centroids
    void assign_range(
        const PointCloud& data,
        std::size_t begin,
        std::size_t end
    ) {
        for (std::size_t row = begin; row < end; ++row) {
            const double* point = &data.values[row * data.dims];

            std::size_t best_cluster = 0;
            double best_distance = std::numeric_limits<double>::max();

            for (std::size_t cluster = 0; cluster < k_; ++cluster) {
                const double distance = squared_distance(
                    point,
                    &centroids_[cluster * data.dims],
                    data.dims
                );

                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }

            labels_[row] = best_cluster;
        }
    }

    // Splits point assignment work across multiple threads
    void assign_clusters_parallel(const PointCloud& data) {
        const std::size_t actual_threads =
            std::min(thread_count_, data.rows);

        const std::size_t block_size =
            (data.rows + actual_threads - 1) / actual_threads;

        std::vector<std::thread> workers;
        workers.reserve(actual_threads);

        for (std::size_t t = 0; t < actual_threads; ++t) {
            const std::size_t begin = t * block_size;
            const std::size_t end =
                std::min(data.rows, begin + block_size);

            if (begin >= end) {
                break;
            }

            workers.emplace_back(
                &ParallelKMeans::assign_range,
                this,
                std::cref(data),
                begin,
                end
            );
        }

        // Wait for every worker thread to finish
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    // Recalculates centroids and returns their total movement
    double recompute_centroids(const PointCloud& data) {
        std::vector<double> sums(k_ * data.dims, 0.0);
        std::vector<std::size_t> counts(k_, 0);

        for (std::size_t row = 0; row < data.rows; ++row) {
            const std::size_t cluster = labels_[row];
            ++counts[cluster];

            for (std::size_t d = 0; d < data.dims; ++d) {
                sums[cluster * data.dims + d] +=
                    data.values[row * data.dims + d];
            }
        }

        double total_shift = 0.0;

        for (std::size_t cluster = 0; cluster < k_; ++cluster) {
            if (counts[cluster] == 0) {
                continue;
            }

            for (std::size_t d = 0; d < data.dims; ++d) {
                const std::size_t index =
                    cluster * data.dims + d;

                const double new_value =
                    sums[index] /
                    static_cast<double>(counts[cluster]);

                const double delta =
                    new_value - centroids_[index];

                total_shift += delta * delta;
                centroids_[index] = new_value;
            }
        }

        return std::sqrt(total_shift);
    }

    std::size_t k_;
    std::size_t max_iterations_;
    double tolerance_;
    std::size_t thread_count_;
    std::vector<double> centroids_;
    std::vector<std::size_t> labels_;
};

int main() {
    try {
        constexpr std::size_t cluster_count = 4;
        constexpr std::size_t dimensions = 8;

        // Generate 50,000 points for each of four clusters
        const PointCloud data =
            make_clustered_data(
                50'000,
                dimensions,
                cluster_count
            );

        // Create and train the K-Means model
        ParallelKMeans model(
            cluster_count,
            50,
            1e-6
        );

        model.fit(data);

        // Print the final centroid for each cluster
        std::cout << "\nFinal centroids:\n";

        const auto& centroids = model.centroids();

        for (std::size_t cluster = 0;
             cluster < cluster_count;
             ++cluster) {
            std::cout << "Cluster " << cluster << ": ";

            for (std::size_t d = 0; d < dimensions; ++d) {
                std::cout << std::fixed
                          << std::setprecision(3)
                          << centroids[
                              cluster * dimensions + d
                          ]
                          << (d + 1 == dimensions ? '\n' : ' ');
            }
        }
    } catch (const std::exception& exception) {
        // Report invalid input or runtime errors
        std::cerr << "Error: "
                  << exception.what()
                  << '\n';

        return 1;
    }

    return 0;
}