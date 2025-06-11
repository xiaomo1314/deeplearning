#pragma once
#pragma once
#include <vector>
#include <random>

namespace utils {

	// 随机初始化权重
	void random_init(std::vector<std::vector<double>>& weights, int input_dim, int output_dim);
	void random_init(std::vector<double>& bias, int dim);

	// 生成范围内的随机数
	double random_double(double min, double max);

}