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

    // ǰ��
    std::vector<double> forward(const std::vector<double>& input);
    // CNNǰ��
    std::vector<std::vector<std::vector<double>>> forward_conv(const std::vector<std::vector<std::vector<double>>>& input);

    // ���򴫲�ѵ��
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs, double lr);

    // ��������ʧ
    double cross_entropy_loss(const std::vector<double>& pred, const std::vector<double>& label);
};