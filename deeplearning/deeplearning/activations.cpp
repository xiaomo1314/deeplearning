#include "activations.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace activations {

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    double relu(double x) {
        return x > 0 ? x : 0;
    }
    double relu_derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double max_elem = *std::max_element(x.begin(), x.end());
        double sum = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_elem);
            sum += result[i];
        }
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }
        return result;
    }

}