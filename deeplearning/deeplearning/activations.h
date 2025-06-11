#pragma once
#pragma once
#include <vector>

namespace activations {

	// Sigmoid
	double sigmoid(double x);
	double sigmoid_derivative(double x);

	// ReLU
	double relu(double x);
	double relu_derivative(double x);

	// Softmax (applies to a vector)
	std::vector<double> softmax(const std::vector<double>& x);

}