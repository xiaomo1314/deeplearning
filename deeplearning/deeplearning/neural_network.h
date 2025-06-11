#pragma once
#pragma once
#include "layer.h"
#include <vector>
#include <memory>

class NeuralNetwork {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    void add_dense(int input_dim, int output_dim, const std::string& act);
    void add_conv(int in_channels, int out_channels, int kernel_size, int stride, const std::string& act);

    // 前向
    std::vector<double> forward(const std::vector<double>& input);
    // CNN前向
    std::vector<std::vector<std::vector<double>>> forward_conv(const std::vector<std::vector<std::vector<double>>>& input);

    // 反向传播训练
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs, double lr);

    // 交叉熵损失
    double cross_entropy_loss(const std::vector<double>& pred, const std::vector<double>& label);
};