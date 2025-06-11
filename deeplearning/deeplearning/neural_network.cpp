#include "neural_network.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include "activations.h"
#include "utils.h"
#include "layer.h"

void NeuralNetwork::add_dense(int input_dim, int output_dim, const std::string& act) {
    layers.push_back(std::make_shared<Layer>(input_dim, output_dim, act));
}

void NeuralNetwork::add_conv(int in_channels, int out_channels, int kernel_size, int stride, const std::string& act) {
    layers.push_back(std::make_shared<Layer>(in_channels, out_channels, kernel_size, stride, act));
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> out = input;
    for (auto& layer : layers)
        if (layer->type == DENSE)
            out = layer->forward(out);
    return out;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs, double lr) {
    int n = (int)X.size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            // 前向
            std::vector<double> out = forward(X[i]);
            // softmax和损失
            std::vector<double> prob = activations::softmax(out);
            loss_sum += cross_entropy_loss(prob, Y[i]);
            // 反向传播
            std::vector<double> grad(prob.size());
            for (size_t j = 0; j < prob.size(); ++j)
                grad[j] = prob[j] - Y[i][j];
            // 从后往前
            for (int l = (int)layers.size() - 1; l >= 0; --l)
                grad = layers[l]->backward(grad, lr);
        }
        std::cout << "Epoch " << epoch << " Loss: " << loss_sum / n << std::endl;
    }
}

double NeuralNetwork::cross_entropy_loss(const std::vector<double>& pred, const std::vector<double>& label) {
    double sum = 0.0;
    for (size_t i = 0; i < pred.size(); ++i)
        sum -= label[i] * std::log(pred[i] + 1e-8);
    return sum;
}