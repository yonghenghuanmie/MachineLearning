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
		//float error = std::rand() % 21 - 10;
		float error = 0;
		float hx = θ₁ * X₁ + θ₂ * X₂ + error;
		train_output[i] = hx;
	}
	return { train_input/*.normalized()*/, train_output };
}

//              i      i
//1/2m * ∑(h(x) - j(x) )²
Eigen::MatrixXf LossFunction(const Eigen::MatrixXf& model, const Eigen::MatrixXf& train_input, const Eigen::MatrixXf& train_output)
{
	auto hx_sub_jx = train_output - train_input * model;
	return hx_sub_jx.array().pow(2) / (train_input.rows() * 2);
}

//                            i      i     i
//θ = θ - α * 1/m * ∑(h(x) - j(x) ) * x
// j    j                                  j
void GradientDescent(Eigen::MatrixXf& model, const Eigen::MatrixXf& train_input, const Eigen::MatrixXf& train_output, float learning_rate, size_t batch_size)
{
	for (size_t i = 0; i < batch_size; i++)
	{
		for (size_t column = 0; column < model.cols(); column++)
		{
			for (size_t train_index = 0; train_index < train_input.rows() && train_index + model.cols() - 1 < train_input.rows(); train_index++)
			{
				auto hx_sub_jx = train_output - train_input * model;
				/*std::cout << hx_sub_jx << "\n\n";
				std::cout << (hx_sub_jx.row(train_index) * train_input.block(train_index, 0, hx_sub_jx.cols(), train_input.cols())).transpose() << "\n\n";*/
				model -= (learning_rate / train_input.rows() * hx_sub_jx.row(train_index) * train_input.block(train_index, 0, hx_sub_jx.cols(), train_input.cols())).transpose();
				std::cout << model << "\n\n";
			}
		}
	}
}

int main()
{
	auto [train_input, train_output] = RandomGenterateTrainSet(10);
	//std::cout << train_input << std::endl;
	//std::cout << train_output << std::endl;

	Eigen::MatrixXf model = Eigen::Vector2f(1, 50);
	float learning_rate = 1;
	int batch_size = 1;
	GradientDescent(model, train_input, train_output, learning_rate, batch_size);
	return 0;
}