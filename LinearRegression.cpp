#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>

Eigen::Matrix2Xf train_input;
Eigen::VectorXf tarin_output;

//h(x)=θ₁x+θ₂c
//θ₁=10,θ₂=150，c=1
void RandomGenterateTrainSet(size_t counts)
{
	train_input.resize(counts, 2);
	tarin_output.resize(counts, Eigen::NoChange);
	for (size_t i = 0; i < counts; i++)
	{
		float x = std::rand() % 100 + 20;
		train_input(i, 0) = x;
		train_input(i, 1) = 1;
		float c = 150;
		float error = std::rand() % 21 - 10;
		float hx = x * 10 + c + error;
		tarin_output[i] = hx;
	}
}

//1/2m * ∑(h(θ)-j(θ))²
Eigen::MatrixXf LossFunction(const Eigen::MatrixXf& model, size_t counts)
{
	Eigen::VectorXf jθ = train_input * model;
	return (tarin_output - jθ).array().pow(2) / (counts * 2);
}

//θ = θ - α * 1/m * ∑(h(θ)-j(θ))²θ
void GradientDescent(Eigen::MatrixXf& model, float learning_rate, size_t counts)
{
	Eigen::MatrixXf results(train_input.rows(), model.cols());
	for (size_t column = 0; column < model.cols(); column++)
		for (size_t row = 0; row < model.rows(); row++)
			results(row, column) = (learning_rate * LossFunction(model, counts) * 2 * model(row, column)).sum();
	model -= results;
}

int main()
{
	RandomGenterateTrainSet(2);
	Eigen::MatrixXf model = Eigen::Vector2f(1, 50);
	float learning_rate = 2;
	int batch_size = 10;
	for (size_t i = 0; i < batch_size; i++)
	{
		GradientDescent(model, learning_rate, train_input.rows());
	}
	std::cout << model(0, 0) << std::endl;
	std::cout << model(1, 0) << std::endl;
	/*std::cout << train_input << std::endl;
	std::cout << tarin_output << std::endl;*/
	return 0;
}