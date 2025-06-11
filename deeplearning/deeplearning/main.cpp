#include <iostream>
#include"neural_network.h"
#include "activations.h"
int main() {
    NeuralNetwork net;
    // ����һ��2��MLP������ά��4�����ز�8�����2����
    net.add_dense(4, 8, "relu");
    net.add_dense(8, 2, "sigmoid");

    // α�����ݣ��������
    std::vector<std::vector<double>> X = { {0,0,0,0}, {0,1,1,0}, {1,0,1,1}, {1,1,0,1} };
    std::vector<std::vector<double>> Y = { {1,0}, {0,1}, {0,1}, {1,0} };

    net.train(X, Y, 200, 0.1);

    // ����
    for (const auto& x : X) {
        auto out = net.forward(x);
        auto prob = activations::softmax(out);
        std::cout << "Pred: ";
        for (auto p : prob) std::cout << p << " ";
        std::cout << std::endl;
    }
    return 0;
}