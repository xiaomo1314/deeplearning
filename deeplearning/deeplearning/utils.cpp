#include "utils.h"

namespace utils {

    void random_init(std::vector<std::vector<double>>& weights, int input_dim, int output_dim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        weights.resize(output_dim, std::vector<double>(input_dim));
        for (auto& row : weights)
            for (auto& w : row)
                w = dis(gen);
    }

    void random_init(std::vector<double>& bias, int dim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        bias.resize(dim);
        for (auto& b : bias)
            b = dis(gen);
    }

    double random_double(double min, double max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

}