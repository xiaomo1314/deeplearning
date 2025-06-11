//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <cstdlib>
//#include <ctime>
//
//using namespace std;
//
//// 激活函数及其导数
//double sigmoid(double x) {
//    return 1.0 / (1.0 + exp(-x));
//}
//
//double sigmoid_derivative(double x) {
//    return x * (1.0 - x);
//}
//
//class NeuralNetwork {
//private:
//    vector<int> layers;  // 网络各层的神经元数量
//    vector<vector<double>> weights;  // 权重矩阵
//    vector<vector<double>> biases;  // 偏置矩阵
//    double learningRate;  // 学习率
//
//public:
//    // 构造函数
//    NeuralNetwork(vector<int> layerSizes, double lr = 0.1) : layers(layerSizes), learningRate(lr) {
//        srand(time(0));
//
//        // 初始化权重和偏置
//        for (int i = 0; i < layers.size() - 1; i++) {
//            // 权重矩阵：weights[i] 是第i层到第i+1层的权重
//            vector<double> w;
//            for (int j = 0; j < layers[i] * layers[i + 1]; j++) {
//                // 随机初始化权重在-1到1之间
//                w.push_back(((double)rand() / RAND_MAX) * 2.0 - 1.0);
//            }
//            weights.push_back(w);
//
//            // 偏置矩阵：biases[i] 是第i+1层的偏置
//            vector<double> b;
//            for (int j = 0; j < layers[i + 1]; j++) {
//                b.push_back(((double)rand() / RAND_MAX) * 2.0 - 1.0);
//            }
//            biases.push_back(b);
//        }
//    }
//
//    // 前向传播
//    vector<double> feedforward(vector<double> input) {
//        vector<double> activation = input;
//
//        for (int i = 0; i < layers.size() - 1; i++) {
//            vector<double> newActivation(layers[i + 1], 0.0);
//
//            // 计算加权输入和激活值
//            for (int j = 0; j < layers[i + 1]; j++) {
//                double z = biases[i][j];  // 先加上偏置
//                for (int k = 0; k < layers[i]; k++) {
//                    z += activation[k] * weights[i][k * layers[i + 1] + j];
//                }
//                newActivation[j] = sigmoid(z);
//            }
//
//            activation = newActivation;
//        }
//
//        return activation;
//    }
//
//    // 反向传播
//    void backpropagation(vector<double> input, vector<double> target) {
//        // 存储各层的激活值
//        vector<vector<double>> activations;
//        activations.push_back(input);
//
//        // 存储各层的加权输入(z值)
//        vector<vector<double>> zs;
//
//        // 前向传播，保存中间值
//        for (int i = 0; i < layers.size() - 1; i++) {
//            vector<double> z(layers[i + 1], 0.0);
//            vector<double> a(layers[i + 1], 0.0);
//
//            for (int j = 0; j < layers[i + 1]; j++) {
//                z[j] = biases[i][j];
//                for (int k = 0; k < layers[i]; k++) {
//                    z[j] += activations[i][k] * weights[i][k * layers[i + 1] + j];
//                }
//                a[j] = sigmoid(z[j]);
//            }
//
//            zs.push_back(z);
//            activations.push_back(a);
//        }
//
//        // 计算输出层的误差
//        vector<double> outputError(layers.back(), 0.0);
//        for (int i = 0; i < layers.back(); i++) {
//            outputError[i] = (target[i] - activations.back()[i]) *
//                sigmoid_derivative(activations.back()[i]);
//        }
//
//        // 误差反向传播
//        vector<vector<double>> errors;
//        errors.push_back(outputError);
//
//        for (int i = layers.size() - 2; i > 0; i--) {
//            vector<double> error(layers[i], 0.0);
//
//            for (int j = 0; j < layers[i]; j++) {
//                double err = 0.0;
//                for (int k = 0; k < layers[i + 1]; k++) {
//                    err += errors.back()[k] * weights[i][j * layers[i + 1] + k];
//                }
//                error[j] = err * sigmoid_derivative(activations[i][j]);
//            }
//
//            errors.push_back(error);
//        }
//
//        // 反转errors，使其与层的顺序一致
//        reverse(errors.begin(), errors.end());
//
//        // 更新权重和偏置
//        for (int i = 0; i < layers.size() - 1; i++) {
//            for (int j = 0; j < layers[i + 1]; j++) {
//                // 更新偏置
//                biases[i][j] += learningRate * errors[i][j];
//
//                // 更新权重
//                for (int k = 0; k < layers[i]; k++) {
//                    weights[i][k * layers[i + 1] + j] +=
//                        learningRate * errors[i][j] * activations[i][k];
//                }
//            }
//        }
//    }
//
//    // 训练网络
//    void train(vector<vector<double>> inputs, vector<vector<double>> targets, int epochs) {
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            double totalError = 0.0;
//
//            for (int i = 0; i < inputs.size(); i++) {
//                vector<double> output = feedforward(inputs[i]);
//
//                // 计算误差
//                double error = 0.0;
//                for (int j = 0; j < output.size(); j++) {
//                    error += pow(targets[i][j] - output[j], 2);
//                }
//                totalError += error;
//
//                // 反向传播
//                backpropagation(inputs[i], targets[i]);
//            }
//
//            // 打印每轮的平均误差
//            if (epoch % 1000 == 0) {
//                cout << "Epoch " << epoch << " error: " << totalError / inputs.size() << endl;
//            }
//        }
//    }
//};
//
//int main() {
//    // 创建一个3-4-2的神经网络（3个输入神经元，4个隐藏神经元，2个输出神经元）
//    NeuralNetwork nn({ 3, 4, 2 }, 0.1);
//
//    // XOR问题的训练数据（扩展为3输入）
//    vector<vector<double>> inputs = {
//        {0, 0, 0},
//        {0, 0, 1},
//        {0, 1, 0},
//        {0, 1, 1},
//        {1, 0, 0},
//        {1, 0, 1},
//        {1, 1, 0},
//        {1, 1, 1}
//    };
//
//    vector<vector<double>> targets = {
//        {0, 0},  // 0 XOR 0 = 0
//        {0, 1},  // 0 XOR 1 = 1
//        {0, 1},  // 1 XOR 0 = 1
//        {1, 0},  // 1 XOR 1 = 0
//        {0, 1},  // 1 XOR 0 = 1
//        {1, 0},  // 1 XOR 1 = 0
//        {1, 0},  // 1 XOR 1 = 0
//        {1, 1}   // 1 XOR 1 + 1 = 1
//    };
//
//    // 训练网络
//    nn.train(inputs, targets, 10000);
//
//    // 测试网络
//    cout << "\n测试结果:" << endl;
//    for (int i = 0; i < inputs.size(); i++) {
//        vector<double> output = nn.feedforward(inputs[i]);
//        cout << "输入: ";
//        for (double val : inputs[i]) {
//            cout << val << " ";
//        }
//        cout << " -> 输出: ";
//        for (double val : output) {
//            cout << val << " ";
//        }
//        cout << endl;
//    }
//
//    return 0;
//}