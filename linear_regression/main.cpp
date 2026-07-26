#include <algorithm>  
#include <cmath>      // Mathematical operations (multiplication and powers)
#include <cstddef>    
#include <iomanip>    
#include <iostream>   
#include <random>     
#include <stdexcept>  
#include <vector>     // std::vector

/*
    Dataset stores the complete feature matrix and target vector.

    Memory layout:

        x = [
            row 0 feature 0,
            row 0 feature 1,
            ...
            row 0 feature cols - 1,

            row 1 feature 0,
            row 1 feature 1,
            ...
        ]

     x represents a two-dimensional matrix, it is stored as one
    contiguous one-dimensional std::vector<double>.

    A value at logical position [row][col] is accessed with:

        x[row * cols + col]

    This row-major representation avoids allocating a separate vector for
    every row. It improves cache locality because the features belonging to
    one observation are stored next to each other in memory.
*/
struct Dataset {
    // Number of observations in the dataset.
    std::size_t rows{};

    // Number of features in each observation.
    std::size_t cols{};

    /*
        Contiguous feature storage.

        Total number of doubles:

            rows * cols

        Each double takes up 8 bytes, so the feature
        memory usage is:

            rows * cols * sizeof(double)
    */
    std::vector<double> x;

    /*
        Target values.

        y[i] contains the target associated with feature row i.

        Total number of doubles:

            rows
    */
    std::vector<double> y;
};

/*
    Generates a synthetic multivariate linear-regression dataset.

    The generated targets follow:

        y = bias + x0*w0 + x1*w1 + ... + noise

    The function returns Dataset by value. Modern C++ normally avoids an
    expensive deep copy here through return-value optimization or move
    semantics. Ownership of the vectors is transferred efficiently to the
    caller.
*/
Dataset make_synthetic_regression(
    std::size_t rows,
    std::size_t cols,
    unsigned int seed = 42
) {
    /*
        A dataset with zero rows or zero columns cannot be used by the model.

        The exception prevents invalid vector sizes and later indexing errors.
    */
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }

    /*
        Mersenne Twister pseudo-random-number generator.

        generator is a local stack object. It stores its internal random state
        directly within the function's stack frame.

        Supplying a fixed seed makes the generated dataset reproducible.
    */
    std::mt19937 generator(seed);

    /*
        Distribution used to generate feature values.

        Mean = 0.0
        Standard deviation = 1.0

        Calling feature_dist(generator) produces one double.
    */
    std::normal_distribution<double> feature_dist(0.0, 1.0);

    /*
        Distribution used to add random noise to the targets.

        Mean = 0.0
        Standard deviation = 0.25

        The corresponding variance is:

            0.25^2 = 0.0625
    */
    std::normal_distribution<double> noise_dist(0.0, 0.25);

    /*
        Allocate a contiguous array containing one true coefficient per
        feature.

        The constructor:

            std::vector<double> true_weights(cols)

        allocates enough dynamic memory for cols doubles and value-initializes
        each element to 0.0.
    */
    std::vector<double> true_weights(cols);

    /*
        Populate the true coefficient vector.

        static_cast<double> converts j + 1 from std::size_t to double before
        multiplication.

        The generated values are:

            0.75, 1.50, 2.25, 3.00, ...
    */
    for (std::size_t j = 0; j < cols; ++j) {
        true_weights[j] = static_cast<double>(j + 1) * 0.75;
    }

    /*
        Compile-time constant bias.

        constexpr allows the compiler to treat this as a constant expression.
        It does not require dynamic allocation.
    */
    constexpr double true_bias = 1.5;

    /*
        dataset initially created with empty vectors.

        At this point:

            data.rows == 0
            data.cols == 0
            data.x.size() == 0
            data.y.size() == 0
    */
    Dataset data;

    // Store the logical dimensions of the feature matrix.
    data.rows = rows;
    data.cols = cols;

    /*
        Resize the feature vector to hold the entire matrix.

        resize(rows * cols):

        1. Allocates contiguous dynamic memory for rows * cols doubles.
        2. Sets the vector's logical size to rows * cols.
        3. Value-initializes every double to 0.0.

        Because the full size is allocated once, the later training-data loop
        can write directly into existing memory without repeatedly growing the
        vector or triggering reallocations.
    */
    data.x.resize(rows * cols);

    /*
        Allocate one target value for every observation.

        Like data.x, this allocation happens once before the generation loop.
    */
    data.y.resize(rows);

    /*
        Generate one observation at a time.

        i identifies the logical row in the feature matrix.
    */
    for (std::size_t i = 0; i < rows; ++i) {
        /*
            Start the target with the true intercept.

            target is a local double stored on the stack. It accumulates the
            dot product for the current row.
        */
        double target = true_bias;

        /*
            Generate each feature for row i.
        */
        for (std::size_t j = 0; j < cols; ++j) {
            /*
                Generate one normally distributed feature value.

                const prevents the value from being reassigned.
            */
            const double value = feature_dist(generator);

            /*
                Convert the logical two-dimensional coordinate [i][j] into a
                one-dimensional row-major offset.

                    offset = i * cols + j

                Example with cols = 5:

                    row 0 begins at index 0
                    row 1 begins at index 5
                    row 2 begins at index 10

                operator[] performs direct unchecked access to the vector's
                contiguous memory.
            */
            data.x[i * cols + j] = value;

            /*
                Accumulate the linear model:

                    target += feature * coefficient

                true_weights[j] is the coefficient associated with feature j.
            */
            target += value * true_weights[j];
        }

        /*
            Store the final target for row i.

            The target consists of:

                true bias
                + feature-weight dot product
                + random noise
        */
        data.y[i] = target + noise_dist(generator);
    }

    /*
        Return the completed dataset.

        In modern C++, the compiler will generally construct the result
        directly in the caller's storage through named return-value
        optimization. If that optimization is not performed, the vectors can
        still be moved instead of copied.

        Moving a vector transfers ownership of its internal memory buffer.
        It does not copy every double.
    */
    return data;
}

class LinearRegression {
public:
    /*
        Construct a linear-regression model.

        feature_count determines how many coefficients the model stores.
    */
    LinearRegression(
        std::size_t feature_count,
        double learning_rate = 0.03,
        std::size_t epochs = 2500
    )
        /*
            Member initializer list.

            weights_(feature_count, 0.0):

            1. Allocates contiguous memory for feature_count doubles.
            2. Initializes every model weight to 0.0.

            learning_rate_ and epochs_ are copied into the corresponding
            scalar data members.
        */
        : weights_(feature_count, 0.0),
          learning_rate_(learning_rate),
          epochs_(epochs) {
        if (feature_count == 0) {
            throw std::invalid_argument("feature_count must be positive");
        }

        if (learning_rate <= 0.0) {
            throw std::invalid_argument("learning_rate must be positive");
        }
    }

    /*
        Train the model using full-batch gradient descent.

        The Dataset is passed by const reference:

            const Dataset& data

        This is important for memory efficiency.

        Passing by value would copy:

            data.x
            data.y
            rows
            cols

        Passing by reference gives the function access to the existing Dataset
        object without copying its large vector buffers.

        const prevents fit() from modifying the caller's dataset.
    */
    void fit(const Dataset& data) {
        /*
            Confirm that the number of dataset columns matches the number of
            model weights.
        */
        if (data.cols != weights_.size()) {
            throw std::invalid_argument(
                "dataset feature count does not match model"
            );
        }

        /*
            Validate the relationship between logical dimensions and physical
            vector sizes.

            For a valid row-major matrix:

                x.size() == rows * cols

            For the target vector:

                y.size() == rows
        */
        if (
            data.rows == 0 ||
            data.x.size() != data.rows * data.cols ||
            data.y.size() != data.rows
        ) {
            throw std::invalid_argument("invalid dataset");
        }

        /*
            Allocate one gradient accumulator per model weight.

            This vector is created once before the epoch loop. Reusing the same
            allocation avoids repeatedly requesting and releasing heap memory
            during training.
        */
        std::vector<double> weight_gradient(weights_.size(), 0.0);

        /*
            Perform the requested number of complete passes over the dataset.
        */
        for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
            /*
                Reset all accumulated weight gradients to zero.

                std::fill writes 0.0 into every existing element. It does not
                change the vector's size or capacity and does not allocate new
                memory.
            */
            std::fill(
                weight_gradient.begin(),
                weight_gradient.end(),
                0.0
            );

            /*
                Scalar gradient accumulator for the bias.

                Stored on the stack and reset once per epoch.
            */
            double bias_gradient = 0.0;

            /*
                Accumulates the sum of squared residuals for reporting MSE.
            */
            double squared_error_sum = 0.0;

            /*
                Iterate through every observation.

                This is full-batch gradient descent because all rows contribute
                to the gradient before the weights are updated.
            */
            for (std::size_t i = 0; i < data.rows; ++i) {
                /*
                    Obtain a raw pointer to the first feature of row i.

                    &data.x[i * data.cols]

                    data.x stores all observations contiguously. The expression
                    i * data.cols computes the offset of row i.

                    The resulting pointer refers directly to memory owned by
                    data.x. No row is copied.

                    const double* means:

                    - The pointer stores the address of a double.
                    - The doubles accessed through the pointer cannot be
                      modified.
                    - Pointer arithmetic such as row[j] accesses subsequent
                      contiguous elements.

                    This avoids creating a temporary vector for every row.
                */
                const double* row = &data.x[i * data.cols];

                /*
                    Initialize the prediction with the current model bias.
                */
                double prediction = bias_;

                /*
                    Compute the dot product:

                        prediction =
                            bias +
                            row[0] * weights_[0] +
                            row[1] * weights_[1] +
                            ...

                    Both row and weights_ are contiguous arrays. Sequential
                    access improves CPU cache behavior and gives the compiler
                    an opportunity to vectorize the loop.
                */
                for (std::size_t j = 0; j < data.cols; ++j) {
                    prediction += row[j] * weights_[j];
                }

                /*
                    Residual for the current observation.

                    A positive error means the prediction is above the target.
                    A negative error means it is below the target.
                */
                const double error = prediction - data.y[i];

                /*
                    accumulate squared error for later MSE calculation.

                    Squaring removes the sign and penalizes larger errors more
                    strongly.
                */
                squared_error_sum += error * error;

                /*
                    The derivative of squared error with respect to the bias is
                    proportional to the residual.

                    Accumulate the contribution from this observation.
                */
                bias_gradient += error;

                /*
                    Accumulate the gradient contribution for every weight.

                    For one observation:

                        d(error^2) / d(weight_j)
                            = 2 * error * feature_j

                    The factor of 2 is applied later through scale.
                */
                for (std::size_t j = 0; j < data.cols; ++j) {
                    weight_gradient[j] += error * row[j];
                }
            }

            /*
                Scale converts accumulated gradient sums into the gradient of
                mean squared error.

                MSE:

                    (1 / n) * sum(error^2)

                Derivative:

                    (2 / n) * sum(error * feature)

                data.rows is converted to double to avoid integer division.
            */
            const double scale =
                2.0 / static_cast<double>(data.rows);

            /*
                Update each weight in place.

                No new weight vector is created. Existing values in the
                weights_ memory buffer are overwritten.

                Gradient descent update:

                    weight =
                        weight
                        - learning_rate
                        * gradient
            */
            for (std::size_t j = 0; j < weights_.size(); ++j) {
                weights_[j] -=
                    learning_rate_ *
                    scale *
                    weight_gradient[j];
            }

            /*
                Update the scalar bias using its accumulated gradient.
            */
            bias_ -=
                learning_rate_ *
                scale *
                bias_gradient;

            /*
                Print progress every 250 epochs and on the final epoch.
            */
            if (epoch % 250 == 0 || epoch + 1 == epochs_) {
                /*
                    Convert the accumulated squared-error sum into mean squared
                    error.

                    This computation uses the predictions generated before the
                    current parameter update. The displayed MSE therefore
                    describes the model state used during that epoch's forward
                    pass.
                */
                const double mse =
                    squared_error_sum /
                    static_cast<double>(data.rows);

                std::cout
                    << "Epoch "
                    << std::setw(4)
                    << epoch
                    << " | MSE: "
                    << std::fixed
                    << std::setprecision(6)
                    << mse
                    << '\n';
            }
        }
    }

    /*
        Predict a target from a raw contiguous feature array.

        features points to the first double in an existing feature sequence.

        This method does not own the pointed-to memory and does not free it.
        The caller must ensure that:

        - The pointer is valid.
        - At least count doubles are available.
        - The pointed-to memory remains alive during this function call.

        const after the function signature means predict() does not modify the
        LinearRegression object.
    */
    double predict(
        const double* features,
        std::size_t count
    ) const {
        if (count != weights_.size()) {
            throw std::invalid_argument("incorrect feature count");
        }

        /*
            Start with the learned bias.
        */
        double result = bias_;

        /*
            Compute a dot product between the feature array and weight vector.

            features[j] uses pointer indexing. Internally, this is equivalent
            to:

                *(features + j)

            No feature values are copied.
        */
        for (std::size_t j = 0; j < count; ++j) {
            result += features[j] * weights_[j];
        }

        return result;
    }

    /*
        Return a read-only reference to the internal weight vector.

        Returning:

            const std::vector<double>&

        avoids copying the entire vector.

        const prevents the caller from modifying the weights through this
        reference.

        The returned reference remains valid only while the LinearRegression
        object remains alive and while the vector is not reallocated.
    */
    const std::vector<double>& weights() const noexcept {
        return weights_;
    }

    /*
        Return the bias by value.

        A double is small, so copying it is inexpensive.

        noexcept states that this function is guaranteed not to throw an
        exception.
    */
    double bias() const noexcept {
        return bias_;
    }

private:
    /*
        Contiguous model coefficient storage.

        weights_[j] corresponds to feature j.
    */
    std::vector<double> weights_;

    /*
        Scalar intercept parameter.

        Brace initialization sets the initial value to 0.0.
    */
    double bias_{0.0};

    // Step size used for each gradient-descent update.
    double learning_rate_;

    // Number of complete passes over the training dataset.
    std::size_t epochs_;
};

/*
    Calculate mean squared error on a dataset.

    Both objects are passed by const reference, so neither the model nor the
    dataset is copied or modified.
*/
double mean_squared_error(
    const LinearRegression& model,
    const Dataset& data
) {
    /*
        Accumulate the sum of squared prediction errors.
    */
    double total = 0.0;

    for (std::size_t i = 0; i < data.rows; ++i) {
        /*
            Pass a pointer to the first element of row i directly into predict.

            &data.x[i * data.cols]

            This avoids allocating or copying a temporary feature vector.
        */
        const double prediction =
            model.predict(
                &data.x[i * data.cols],
                data.cols
            );

        const double error =
            prediction - data.y[i];

        total += error * error;
    }

    /*
        Divide by the number of observations.

        static_cast<double> prevents integer division.
    */
    return total / static_cast<double>(data.rows);
}

int main() {
    try {
        /*
            Compile-time feature count.

            std::size_t is used because it matches container sizes and indices.
        */
        constexpr std::size_t feature_count = 5;

        /*
            Construct the training dataset.

            The returned Dataset owns two dynamically allocated buffers:

                training_data.x
                training_data.y

            const prevents accidental modification after creation.
        */
        const Dataset training_data =
            make_synthetic_regression(
                20'000,
                feature_count,
                42
            );

        /*
            Construct an independent test dataset with a different seed.

            Because the seed differs, the observations and noise values differ
            from the training data while following the same underlying linear
            relationship.
        */
        const Dataset test_data =
            make_synthetic_regression(
                5'000,
                feature_count,
                1337
            );

        /*
            Construct the model.

            This allocates a weight vector containing five doubles initialized
            to zero.
        */
        LinearRegression model(
            feature_count,
            0.03,
            2500
        );

        /*
            Pass training_data by const reference.

            The large feature and target vectors are not copied.
        */
        model.fit(training_data);

        std::cout << "\nLearned weights:\n";

        /*
            model.weights() returns a const reference to the model's existing
            vector.

            No vector copy is created by the accessor. However, calling the
            function repeatedly inside the loop is unnecessary. It is valid,
            but the reference could also be stored once:

                const auto& learned_weights = model.weights();
        */
        for (
            std::size_t i = 0;
            i < model.weights().size();
            ++i
        ) {
            std::cout
                << "w"
                << i
                << " = "
                << std::fixed
                << std::setprecision(4)
                << model.weights()[i]
                << '\n';
        }

        std::cout
            << "bias = "
            << model.bias()
            << '\n';

        /*
            Evaluate the trained model using the independent test data.

            model and test_data are passed by const reference into
            mean_squared_error, avoiding copies.
        */
        std::cout
            << "test MSE = "
            << mean_squared_error(model, test_data)
            << '\n';
    } catch (const std::exception& exception) {
        /*
            Catch any standard exception thrown by dataset validation,
            construction, or training.

            exception is received by const reference, so the exception object
            is not copied.
        */
        std::cerr
            << "Error: "
            << exception.what()
            << '\n';

        return 1;
    }

    /*
        When main exits, automatic objects are destroyed in reverse order:

            model
            test_data
            training_data

        Each std::vector destructor automatically releases its dynamically
        allocated memory. No manual delete[] calls are required.
    */
    return 0;
}
