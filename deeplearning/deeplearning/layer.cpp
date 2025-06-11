#include "layer.h"
#include "activations.h"
#include "utils.h"
#include <cassert>

Layer::Layer(int input_dim, int output_dim, const std::string& act) {
    type = DENSE;
    activation = act;
    utils::random_init(weights, input_dim, output_dim);
    utils::random_init(bias, output_dim);
}

Layer::Layer(int in_c, int out_c, int k, int s, const std::string& act) {
    type = CONV;
    in_channels = in_c;
    out_channels = out_c;
    kernel_size = k;
    stride = s;
    activation = act;
    kernels.resize(out_c, std::vector<std::vector<double>>(in_c, std::vector<double>(k * k)));
    for (auto& oc : kernels)
        for (auto& ic : oc)
            for (auto& w : ic)
                w = utils::random_double(-0.1, 0.1);
    conv_bias.resize(out_c, 0.0);
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    last_input = input;
    int out_dim = (int)bias.size();
    std::vector<double> z(out_dim, 0.0);
    for (int i = 0; i < out_dim; ++i) {
        z[i] = bias[i];
        for (size_t j = 0; j < input.size(); ++j)
            z[i] += weights[i][j] * input[j];
    }
    last_z = z; // 保存激活前
    std::vector<double> output(out_dim, 0.0);
    for (int i = 0; i < out_dim; ++i) {
        if (activation == "relu")
            output[i] = activations::relu(z[i]);
        else if (activation == "sigmoid")
            output[i] = activations::sigmoid(z[i]);
        else
            output[i] = z[i];
    }
    last_output = output;
    return output;
}

std::vector<std::vector<std::vector<double>>> Layer::forward_conv(const std::vector<std::vector<std::vector<double>>>& input) {
    last_input_conv = input;
    int H = (int)input[0].size(), W = (int)input[0][0].size();
    int out_h = (H - kernel_size) / stride + 1;
    int out_w = (W - kernel_size) / stride + 1;
    std::vector<std::vector<std::vector<double>>> output(out_channels, std::vector<std::vector<double>>(out_h, std::vector<double>(out_w, 0.0)));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                double sum = conv_bias[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            sum += input[ic][oh * stride + kh][ow * stride + kw] * kernels[oc][ic][kh * kernel_size + kw];
                        }
                    }
                }
                // 激活                
                if (activation == "relu")
                    sum = activations::relu(sum);
                else if (activation == "sigmoid")
                    sum = activations::sigmoid(sum);
                output[oc][oh][ow] = sum;
            }
        }
    }
    last_output_conv = output;
    return output;
}

std::vector<double> Layer::backward(const std::vector<double>& grad, double lr) {
    assert(grad.size() == last_z.size());
    std::vector<double> grad_input(last_input.size(), 0.0);
    std::vector<double> grad_z(grad.size());
    for (size_t i = 0; i < grad.size(); ++i) {
        if (activation == "relu")
            grad_z[i] = grad[i] * activations::relu_derivative(last_z[i]);
        else if (activation == "sigmoid")
            grad_z[i] = grad[i] * activations::sigmoid_derivative(last_z[i]);
        else
            grad_z[i] = grad[i];
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            grad_input[j] += weights[i][j] * grad_z[i];
            weights[i][j] -= lr * grad_z[i] * last_input[j];
        }
        bias[i] -= lr * grad_z[i];
    }
    return grad_input;
}