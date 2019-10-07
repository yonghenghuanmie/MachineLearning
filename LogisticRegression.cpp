#include <cmath>
#include <cstdlib>
#include <tuple>
#include <iostream>
#include <type_traits>
#include <Eigen/Dense>
//#include "lazy.h"

namespace LogisticRegression
{
	//Z(X₁,X₂)=θ₁X₁+θ₂X₂
	//θ₁=10,θ₂=150,X₂=1
	std::tuple<Eigen::MatrixX2d, Eigen::VectorXd> RandomGenterateTrainSet(size_t counts)
	{
		Eigen::MatrixX2d train_input(counts, 2);
		Eigen::VectorXd train_output(counts);
		for (size_t i = 0; i < counts; i++)
		{
			float X₁ = std::rand() % 100 + 20, X₂ = 1;
			train_input(i, 0) = X₁;
			train_input(i, 1) = X₂;
			float θ₁ = 10, θ₂ = 150;
			float error = std::rand() % 11 - 5;
			float hx = θ₁ * X₁ + θ₂ * X₂ + error;
			train_output[i] = hx > 800 ? 1 : 0;
		}
		return { (train_input.normalized().array() - 0.5).matrix(), train_output };
	}

	//        -z
	//hx=1/1+e
	//          i      i      i         i
	//-1/m * ∑y log(hx )+(1-y )log(1-hx )
	double LossFunction(const Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
	{
		auto hx = 1 / (1 + Eigen::exp(-1 * (train_input * model).array()));
		std::cout << hx << std::endl;
		return -1.0 / train_output.rows() * (train_output.array() * Eigen::log(hx) + (1 - train_output.array()) * Eigen::log(1 - hx)).sum();
	}
}

int main()
{
	std::cout << std::boolalpha;

	auto [train_input, train_output] = LogisticRegression::RandomGenterateTrainSet(100);
	//std::cout << train_input << std::endl;
	//std::cout << train_output << std::endl;

	Eigen::MatrixXd model = Eigen::Vector2d(10, 100);
	Eigen::Vector2d learning_rate(0.001, 1);
	double limit = 0.05;
	size_t batch_size = 25;
	std::cout << LogisticRegression::LossFunction(model, train_input, train_output) << std::endl;
	return 0;
}