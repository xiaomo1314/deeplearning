#pragma once
#include <vector>
#include <string>
enum LayerType { DENSE, CONV };
class Layer {
public:
    LayerType type;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    int kernel_size = 0;
    int stride = 1;
    int out_channels = 0;
    std::vector<std::vector<std::vector<double>>> kernels; // [out_c][in_c][k*k]
    std::vector<double> conv_bias;
    int in_channels = 0;
    std::string activation;
    std::vector<double> last_input;
    std::vector<double> last_z;
    std::vector<double> last_output;
    std::vector<std::vector<std::vector<double>>> last_input_conv;
    std::vector<std::vector<std::vector<double>>> last_z_conv;
    std::vector<std::vector<std::vector<double>>> last_output_conv;
    Layer(int input_dim, int output_dim, const std::string& act);
    Layer(int in_channels, int out_channels, int kernel_size, int stride, const std::string& act);
    std::vector<double> forward(const std::vector<double>& input);
    std::vector<std::vector<std::vector<double>>> forward_conv(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<double> backward(const std::vector<double>& grad, double lr);
};