#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Activation function used by a dense layer
enum class Activation {
    ReLU,
    None
};

class DenseLayer {
public:
    // Creates a fully connected layer with the requested dimensions
    DenseLayer(
        std::size_t input_size,
        std::size_t output_size,
        Activation activation
    )
        : input_size_(input_size),
          output_size_(output_size),
          weights_(input_size * output_size),
          biases_(output_size),
          activation_(activation) {
        if (input_size == 0 || output_size == 0) {
            throw std::invalid_argument("layer sizes must be positive");
        }
    }

    // Initializes weights with Xavier initialization and biases with zero
    void randomize(std::mt19937& generator) {
        const double limit =
            std::sqrt(6.0 / static_cast<double>(input_size_ + output_size_));
        std::uniform_real_distribution<double> distribution(-limit, limit);

        for (double& weight : weights_) {
            weight = distribution(generator);
        }

        std::fill(biases_.begin(), biases_.end(), 0.0);
    }

    // Runs one forward pass through this layer
    std::vector<double> forward(
        const std::vector<double>& input
    ) const {
        if (input.size() != input_size_) {
            throw std::invalid_argument("input size mismatch");
        }

        std::vector<double> output(output_size_, 0.0);

        for (std::size_t out = 0; out < output_size_; ++out) {
            // Locate the row of weights for the current output neuron
            const double* weight_row =
                &weights_[out * input_size_];

            double sum = biases_[out];

            // Compute the weighted sum for this neuron
            for (std::size_t in = 0; in < input_size_; ++in) {
                sum += weight_row[in] * input[in];
            }

            // Apply ReLU when the layer uses it
            if (activation_ == Activation::ReLU) {
                sum = std::max(0.0, sum);
            }

            output[out] = sum;
        }

        return output;
    }

    // Writes the layer structure and parameters to a binary stream
    void save(std::ofstream& stream) const {
        stream.write(
            reinterpret_cast<const char*>(&input_size_),
            sizeof(input_size_)
        );

        stream.write(
            reinterpret_cast<const char*>(&output_size_),
            sizeof(output_size_)
        );

        const int activation_value =
            static_cast<int>(activation_);

        stream.write(
            reinterpret_cast<const char*>(&activation_value),
            sizeof(activation_value)
        );

        stream.write(
            reinterpret_cast<const char*>(weights_.data()),
            static_cast<std::streamsize>(
                weights_.size() * sizeof(double)
            )
        );

        stream.write(
            reinterpret_cast<const char*>(biases_.data()),
            static_cast<std::streamsize>(
                biases_.size() * sizeof(double)
            )
        );
    }

    // Reconstructs a layer from a binary stream
    static DenseLayer load(std::ifstream& stream) {
        std::size_t input_size = 0;
        std::size_t output_size = 0;
        int activation_value = 0;

        stream.read(
            reinterpret_cast<char*>(&input_size),
            sizeof(input_size)
        );

        stream.read(
            reinterpret_cast<char*>(&output_size),
            sizeof(output_size)
        );

        stream.read(
            reinterpret_cast<char*>(&activation_value),
            sizeof(activation_value)
        );

        DenseLayer layer(
            input_size,
            output_size,
            static_cast<Activation>(activation_value)
        );

        stream.read(
            reinterpret_cast<char*>(layer.weights_.data()),
            static_cast<std::streamsize>(
                layer.weights_.size() * sizeof(double)
            )
        );

        stream.read(
            reinterpret_cast<char*>(layer.biases_.data()),
            static_cast<std::streamsize>(
                layer.biases_.size() * sizeof(double)
            )
        );

        if (!stream) {
            throw std::runtime_error("failed to read layer");
        }

        return layer;
    }

private:
    std::size_t input_size_;
    std::size_t output_size_;
    std::vector<double> weights_;
    std::vector<double> biases_;
    Activation activation_;
};

class NeuralNetwork {
public:
    // Adds a dense layer to the end of the network
    void add(DenseLayer layer) {
        layers_.push_back(std::move(layer));
    }

    // Passes the input through all layers and returns class probabilities
    std::vector<double> predict(
        const std::vector<double>& input
    ) const {
        std::vector<double> activation = input;

        for (const DenseLayer& layer : layers_) {
            activation = layer.forward(activation);
        }

        return softmax(activation);
    }

    // Saves the complete network to a binary file
    void save(const std::string& path) const {
        std::ofstream stream(path, std::ios::binary);

        if (!stream) {
            throw std::runtime_error("could not open model file");
        }

        const std::size_t layer_count = layers_.size();

        stream.write(
            reinterpret_cast<const char*>(&layer_count),
            sizeof(layer_count)
        );

        for (const DenseLayer& layer : layers_) {
            layer.save(stream);
        }
    }

    // Loads a complete network from a binary file
    static NeuralNetwork load(const std::string& path) {
        std::ifstream stream(path, std::ios::binary);

        if (!stream) {
            throw std::runtime_error("could not open model file");
        }

        std::size_t layer_count = 0;

        stream.read(
            reinterpret_cast<char*>(&layer_count),
            sizeof(layer_count)
        );

        NeuralNetwork network;

        for (std::size_t i = 0; i < layer_count; ++i) {
            network.add(DenseLayer::load(stream));
        }

        return network;
    }

private:
    // Converts output logits into normalized class probabilities
    static std::vector<double> softmax(
        const std::vector<double>& logits
    ) {
        if (logits.empty()) {
            throw std::invalid_argument("softmax input cannot be empty");
        }

        // Subtract the maximum logit for numerical stability
        const double maximum =
            *std::max_element(logits.begin(), logits.end());

        std::vector<double> probabilities(logits.size());
        double sum = 0.0;

        // Exponentiate each stabilized logit.
        for (std::size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] = std::exp(logits[i] - maximum);
            sum += probabilities[i];
        }

        // Normalize values so the probabilities sum to one.
        for (double& value : probabilities) {
            value /= sum;
        }

        return probabilities;
    }

    std::vector<DenseLayer> layers_;
};

int main() {
    try {
        // fixed seed keeps the random initialization reproducible
        std::mt19937 generator(42);

        // initializing a 16 → 32 → 16 → 4 network
        DenseLayer hidden_1(16, 32, Activation::ReLU);
        DenseLayer hidden_2(32, 16, Activation::ReLU);
        DenseLayer output(16, 4, Activation::None);

        // Initialize each layer's parameters
        hidden_1.randomize(generator);
        hidden_2.randomize(generator);
        output.randomize(generator);

        // Assemble the network in forward order
        NeuralNetwork network;
        network.add(std::move(hidden_1));
        network.add(std::move(hidden_2));
        network.add(std::move(output));

        // nn input with 16 features
        const std::vector<double> sample = {
            0.21, -0.15, 0.90, 0.40,
            0.10,  0.35, 0.18, 0.77,
           -0.32,  0.14, 0.55, 0.61,
            0.09, -0.42, 0.28, 0.49
        };

        // Generate class probabilities
        const std::vector<double> probabilities =
            network.predict(sample);

        std::cout << "Class probabilities:\n";

        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::cout << "class " << i << ": "
                      << std::fixed << std::setprecision(6)
                      << probabilities[i] << '\n';
        }

        // Find the class with the highest probability
        const auto best = std::max_element(
            probabilities.begin(),
            probabilities.end()
        );

        std::cout << "predicted class: "
                  << std::distance(probabilities.begin(), best)
                  << '\n';

        // Save and restore the model
        network.save("model.bin");

        const NeuralNetwork restored =
            NeuralNetwork::load("model.bin");

        const auto restored_probabilities =
            restored.predict(sample);

        // Verify that the restored model produces the same result
        std::cout << "restored model first probability: "
                  << restored_probabilities.front() << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}