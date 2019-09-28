#include <cstdlib>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

//h(X₁,X₂)=θ₁X₁+θ₂X₂
//θ₁=10,θ₂=150,X₂=1
std::tuple<Eigen::MatrixX2f, Eigen::VectorXf> RandomGenterateTrainSet(size_t counts)
{
	Eigen::MatrixX2f train_input;
	Eigen::VectorXf train_output;
	train_input.resize(counts, Eigen::NoChange);
	train_output.resize(counts, Eigen::NoChange);
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
	return { train_input.normalized(), train_output };
}

//1/2m * ∑(h(x)-j(x))²
Eigen::MatrixXf LossFunction(const Eigen::MatrixXf& model, const Eigen::MatrixXf& train_input, const Eigen::MatrixXf& train_output)
{
	Eigen::VectorXf jx = train_input * model;
	return (train_output - jx).array().pow(2) / (train_input.rows() * 2);
}

//θ = θ - α * 1/m * ∑(h(x)-j(x)) * x
void GradientDescent(Eigen::MatrixXf& model, const Eigen::MatrixXf& train_input, const Eigen::MatrixXf& train_output, float learning_rate, size_t batch_size)
{
	auto hx_sub_jx = train_output - train_input * model;
	for (size_t i = 0; i < batch_size; i++)
	{
		Eigen::MatrixXf results(model.rows(), model.cols());
		for (size_t column = 0; column < model.cols(); column++)
		{
			for (size_t row = 0; row < model.rows(); row++)
			{
				//std::cout << (hx_sub_jx * train_input.row(row)).sum() << std::endl;
				results(row, column) = learning_rate / train_input.rows() * (hx_sub_jx * train_input.row(row)).sum();
			}
		}
		//std::cout << results << std::endl;
		model -= results;
		std::cout << model << std::endl;
	}
}

int main()
{
	auto [train_input, train_output] = RandomGenterateTrainSet(10);
	//std::cout << train_input << std::endl;
	//std::cout << train_output << std::endl;


	Eigen::MatrixXf model = Eigen::Vector2f(1, 50);
	float learning_rate = 1;
	int batch_size = 10;
	GradientDescent(model, train_input, train_output, learning_rate, batch_size);
	return 0;
}