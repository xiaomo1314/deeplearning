//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <cstdlib>
//#include <ctime>
//
//using namespace std;
//
//// ��������䵼��
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
//    vector<int> layers;  // ����������Ԫ����
//    vector<vector<double>> weights;  // Ȩ�ؾ���
//    vector<vector<double>> biases;  // ƫ�þ���
//    double learningRate;  // ѧϰ��
//
//public:
//    // ���캯��
//    NeuralNetwork(vector<int> layerSizes, double lr = 0.1) : layers(layerSizes), learningRate(lr) {
//        srand(time(0));
//
//        // ��ʼ��Ȩ�غ�ƫ��
//        for (int i = 0; i < layers.size() - 1; i++) {
//            // Ȩ�ؾ���weights[i] �ǵ�i�㵽��i+1���Ȩ��
//            vector<double> w;
//            for (int j = 0; j < layers[i] * layers[i + 1]; j++) {
//                // �����ʼ��Ȩ����-1��1֮��
//                w.push_back(((double)rand() / RAND_MAX) * 2.0 - 1.0);
//            }
//            weights.push_back(w);
//
//            // ƫ�þ���biases[i] �ǵ�i+1���ƫ��
//            vector<double> b;
//            for (int j = 0; j < layers[i + 1]; j++) {
//                b.push_back(((double)rand() / RAND_MAX) * 2.0 - 1.0);
//            }
//            biases.push_back(b);
//        }
//    }
//
//    // ǰ�򴫲�
//    vector<double> feedforward(vector<double> input) {
//        vector<double> activation = input;
//
//        for (int i = 0; i < layers.size() - 1; i++) {
//            vector<double> newActivation(layers[i + 1], 0.0);
//
//            // �����Ȩ����ͼ���ֵ
//            for (int j = 0; j < layers[i + 1]; j++) {
//                double z = biases[i][j];  // �ȼ���ƫ��
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
//    // ���򴫲�
//    void backpropagation(vector<double> input, vector<double> target) {
//        // �洢����ļ���ֵ
//        vector<vector<double>> activations;
//        activations.push_back(input);
//
//        // �洢����ļ�Ȩ����(zֵ)
//        vector<vector<double>> zs;
//
//        // ǰ�򴫲��������м�ֵ
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
//        // �������������
//        vector<double> outputError(layers.back(), 0.0);
//        for (int i = 0; i < layers.back(); i++) {
//            outputError[i] = (target[i] - activations.back()[i]) *
//                sigmoid_derivative(activations.back()[i]);
//        }
//
//        // ���򴫲�
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
//        // ��תerrors��ʹ������˳��һ��
//        reverse(errors.begin(), errors.end());
//
//        // ����Ȩ�غ�ƫ��
//        for (int i = 0; i < layers.size() - 1; i++) {
//            for (int j = 0; j < layers[i + 1]; j++) {
//                // ����ƫ��
//                biases[i][j] += learningRate * errors[i][j];
//
//                // ����Ȩ��
//                for (int k = 0; k < layers[i]; k++) {
//                    weights[i][k * layers[i + 1] + j] +=
//                        learningRate * errors[i][j] * activations[i][k];
//                }
//            }
//        }
//    }
//
//    // ѵ������
//    void train(vector<vector<double>> inputs, vector<vector<double>> targets, int epochs) {
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            double totalError = 0.0;
//
//            for (int i = 0; i < inputs.size(); i++) {
//                vector<double> output = feedforward(inputs[i]);
//
//                // �������
//                double error = 0.0;
//                for (int j = 0; j < output.size(); j++) {
//                    error += pow(targets[i][j] - output[j], 2);
//                }
//                totalError += error;
//
//                // ���򴫲�
//                backpropagation(inputs[i], targets[i]);
//            }
//
//            // ��ӡÿ�ֵ�ƽ�����
//            if (epoch % 1000 == 0) {
//                cout << "Epoch " << epoch << " error: " << totalError / inputs.size() << endl;
//            }
//        }
//    }
//};
//
//int main() {
//    // ����һ��3-4-2�������磨3��������Ԫ��4��������Ԫ��2�������Ԫ��
//    NeuralNetwork nn({ 3, 4, 2 }, 0.1);
//
//    // XOR�����ѵ�����ݣ���չΪ3���룩
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
//    // ѵ������
//    nn.train(inputs, targets, 10000);
//
//    // ��������
//    cout << "\n���Խ��:" << endl;
//    for (int i = 0; i < inputs.size(); i++) {
//        vector<double> output = nn.feedforward(inputs[i]);
//        cout << "����: ";
//        for (double val : inputs[i]) {
//            cout << val << " ";
//        }
//        cout << " -> ���: ";
//        for (double val : output) {
//            cout << val << " ";
//        }
//        cout << endl;
//    }
//
//    return 0;
//}