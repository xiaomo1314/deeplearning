#include <iostream>
#include"neural_network.h"
#include "activations.h"
int main() {
    NeuralNetwork net;
    // 构建一个2层MLP，输入维度4，隐藏层8，输出2分类
    net.add_dense(4, 8, "relu");
    net.add_dense(8, 2, "sigmoid");

    // 伪造数据：异或任务
    std::vector<std::vector<double>> X = { {0,0,0,0}, {0,1,1,0}, {1,0,1,1}, {1,1,0,1} };
    std::vector<std::vector<double>> Y = { {1,0}, {0,1}, {0,1}, {1,0} };

    net.train(X, Y, 200, 0.1);

    // 测试
    for (const auto& x : X) {
        auto out = net.forward(x);
        auto prob = activations::softmax(out);
        std::cout << "Pred: ";
        for (auto p : prob) std::cout << p << " ";
        std::cout << std::endl;
    }
    return 0;
}