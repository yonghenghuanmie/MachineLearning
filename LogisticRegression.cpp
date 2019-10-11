#include <cmath>
#include <cstdlib>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

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
		for (size_t i = 0; i < train_input.cols() - 1; i++)
		{
			auto&& col_i = train_input.col(i);
			col_i = (col_i.array() - col_i.sum() / train_input.rows() / 2) / (col_i.maxCoeff() - col_i.minCoeff());
		}
		return { train_input, train_output };
	}

	//        -z
	//hx=1/1+e
	//          i      i      i         i
	//-1/m * ∑y log(hx )+(1-y )log(1-hx )
	double LossFunction(const Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
	{
		auto hx = 1 / (1 + Eigen::exp(-1 * (train_input * model).array()));
		//std::cout << hx << std::endl;
		//std::cout << Eigen::log(1 - hx) << std::endl;
		return -1.0 / train_output.rows() * (train_output.array() * Eigen::log(hx) + (1 - train_output.array()) * Eigen::log(1 - hx)).sum();
	}

	//                            i      i     i
	//θ = θ - α * 1/m * ∑(h(x) - y(x) ) * x
	// j    j                                  j
	void GradientDescent(Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output, const Eigen::VectorXd& learning_rate, size_t batch_size)
	{
		for (size_t i = 0; i < batch_size; i++)
		{
			for (size_t train_index = 0; train_index < train_input.rows(); train_index++)
			{
				auto hx = 1 / (1 + Eigen::exp(-1 * (train_input * model).array()));
				auto hx_sub_yx = hx.matrix() - train_output;
				//std::cout << hx_sub_yx << "\n\n";
				Eigen::MatrixXd duplicate_line(model.cols(), train_input.cols());
				auto&& reference = duplicate_line << train_input.row(train_index);
				for (size_t line = 0; line < model.cols() - 1; line++)
				{
					reference, train_input.row(train_index);
				}
				Eigen::MatrixXd update = learning_rate.array() * (1.0 / train_input.rows() * hx_sub_yx.row(train_index) * duplicate_line).transpose().array();
				//std::cout << duplicate_line << "\n\n";
				std::cout << update << "\n\n";
				std::cout << LossFunction(model, train_input, train_output) << "\n\n";
				model -= update;
				std::cout << model << "\n\n";
			}
		}
	}
}

int main()
{
	std::cout << std::boolalpha;

	auto [train_input, train_output] = LogisticRegression::RandomGenterateTrainSet(100);
	//std::cout << train_input << std::endl;
	//std::cout << train_output << std::endl;

	Eigen::MatrixXd model = Eigen::Vector2d(1, 200);
	Eigen::Vector2d learning_rate(1, 10);
	double limit = 0.05;
	size_t batch_size = 10;
	//std::cout << LogisticRegression::LossFunction(model, train_input, train_output) << std::endl;
	LogisticRegression::GradientDescent(model, train_input, train_output, learning_rate, batch_size);
	return 0;
}