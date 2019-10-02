#include <cstdlib>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

//h(X₁,X₂)=θ₁X₁+θ₂X₂
//θ₁=10,θ₂=150,X₂=1
std::tuple<Eigen::MatrixX2d, Eigen::VectorXd> RandomGenterateTrainSet(size_t counts)
{
	Eigen::MatrixX2d train_input;
	Eigen::VectorXd train_output;
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
Eigen::MatrixXd LossFunction(const Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
{
	auto hx_sub_jx = train_output - train_input * model;
	//not sure
	return hx_sub_jx.array().pow(2) / (train_input.rows() * 2);
}

//                            i      i     i
//θ = θ - α * 1/m * ∑(h(x) - j(x) ) * x
// j    j                                  j
//θ = θ - ? +
void GradientDescent(Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output, float learning_rate, size_t batch_size)
{
	for (size_t i = 0; i < batch_size; i++)
	{
		for (size_t column = 0; column < model.cols(); column++)
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
				/*std::cout << duplicate_line << "\n\n";
				std::cout << (learning_rate / train_input.rows() * hx_sub_jx.row(train_index) * duplicate_line).transpose() << "\n\n";*/
				model += (learning_rate / train_input.rows() * hx_sub_jx.row(train_index) * duplicate_line).transpose();
				std::cout << model << "\n\n";
			}
		}
	}
}

int main()
{
	auto [train_input, train_output] = RandomGenterateTrainSet(10);
	std::cout << train_input << std::endl;
	std::cout << train_output << std::endl;

	/*Eigen::MatrixXd model = Eigen::Vector2d(1, 10);
	float learning_rate = 0.001;
	int batch_size = 10000;*/
	Eigen::MatrixXd model = Eigen::Vector2d(1, 100);
	float learning_rate = 0.001;
	int batch_size = 100;
	GradientDescent(model, train_input, train_output, learning_rate, batch_size);
	return 0;
}