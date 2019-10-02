﻿#include <cstdlib>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

//h(X₁,X₂)=θ₁X₁+θ₂X₂
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
		float error = std::rand() % 21 - 10;
		float hx = θ₁ * X₁ + θ₂ * X₂ + error;
		train_output[i] = hx;
	}
	return { train_input/*.normalized()*/, train_output };
}

//              i      i
//1/2m * ∑(h(x) - j(x) )²
Eigen::MatrixXd LossFunction(const Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
{
	auto hx_sub_jx = train_output - train_input * model;
	return hx_sub_jx.array().pow(2) / (train_input.rows() * 2);
}

//                            i      i     i
//θ = θ - α * 1/m * ∑(h(x) - j(x) ) * x
// j    j                                  j
void GradientDescent(Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output, const Eigen::VectorXd& learning_rate, size_t batch_size)
{
	for (size_t i = 0; i < batch_size; i++)
	{
		for (size_t train_index = 0; train_index < train_input.rows(); train_index++)
		{
			auto hx_sub_jx = train_output - train_input * model;
			//std::cout << hx_sub_jx << "\n\n";
			Eigen::MatrixXd duplicate_line(model.cols(), train_input.cols());
			auto&& reference = duplicate_line << train_input.row(train_index);
			for (size_t line = 0; line < model.cols() - 1; line++)
			{
				reference, train_input.row(train_index);
			}
			Eigen::MatrixXd update = learning_rate.array() * (1.0 / train_input.rows() * hx_sub_jx.row(train_index) * duplicate_line).transpose().array();
			/*std::cout << duplicate_line << "\n\n";
			std::cout << update << "\n\n";*/
			model += update;
			std::cout << model << "\n\n";
		}
	}
}

int main()
{
	auto [train_input, train_output] = RandomGenterateTrainSet(10);
	/*std::cout << train_input << std::endl;
	std::cout << train_output << std::endl;*/

	Eigen::MatrixXd model = Eigen::Vector2d(1, 10);
	Eigen::Vector2d learning_rate(0.001, 0.25);
	int batch_size = 200;
	GradientDescent(model, train_input, train_output, learning_rate, batch_size);
	return 0;
}