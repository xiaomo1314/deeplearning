#pragma once
#pragma once
#include <vector>
#include <random>

namespace utils {

	// �����ʼ��Ȩ��
	void random_init(std::vector<std::vector<double>>& weights, int input_dim, int output_dim);
	void random_init(std::vector<double>& bias, int dim);

	// ���ɷ�Χ�ڵ������
	double random_double(double min, double max);

}