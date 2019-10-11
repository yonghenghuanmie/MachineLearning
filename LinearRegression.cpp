#include <cstdlib>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace LinearRegression
{
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
			float error = std::rand() % 11 - 5;
			float hx = θ₁ * X₁ + θ₂ * X₂ + error;
			train_output[i] = hx;
		}
		return { train_input/*.normalized()*/, train_output };
	}

	//              i      i
	//1/2m * ∑(h(x) - y(x) )²
	double LossFunction(const Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
	{
		auto hx_sub_jx = train_output - train_input * model;
		return hx_sub_jx.array().pow(2).sum() / (train_input.rows() * 2);
	}

	//                            i      i     i
	//θ = θ - α * 1/m * ∑(h(x) - y(x) ) * x
	// j    j                                  j
	void GradientDescent(Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output, const Eigen::VectorXd& learning_rate, double limit)
	{
		size_t batch_size = 0;
		Eigen::MatrixXd update = Eigen::MatrixXd::Zero(model.rows(), model.cols());
		do
		{
			batch_size++;
			Eigen::MatrixXd last_update;
			for (size_t train_index = 0; train_index < train_input.rows(); train_index++)
			{
				auto hx_sub_yx = train_input * model - train_output;
				//std::cout << hx_sub_yx << "\n\n";
				Eigen::MatrixXd duplicate_line(model.cols(), train_input.cols());
				auto&& reference = duplicate_line << train_input.row(train_index);
				for (size_t line = 0; line < model.cols() - 1; line++)
				{
					reference, train_input.row(train_index);
				}
				last_update = update;
				update = learning_rate.array() * (1.0 / train_input.rows() * hx_sub_yx.row(train_index) * duplicate_line).transpose().array();
				//std::cout << duplicate_line << "\n\n";
				//std::cout << update << "\n\n";
				model -= update;
				std::cout << model << "\n\n";
			}

			if (last_update.minCoeff() != 0)
			{
				//non-convergence
				if ((update.array().abs() > last_update.array().abs()).sum())
					break;
				//convergence
				if ((update.array() / last_update.array()).abs().maxCoeff() < limit)
					break;
			}
		} while (true);
		std::cout << "batch_size:" << batch_size << "\n\n";
	}

	void GradientDescent(Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output, const Eigen::VectorXd& learning_rate, size_t batch_size)
	{
		for (size_t i = 0; i < batch_size; i++)
		{
			for (size_t train_index = 0; train_index < train_input.rows(); train_index++)
			{
				auto hx_sub_yx = train_input * model - train_output;
				//std::cout << hx_sub_yx << "\n\n";
				Eigen::MatrixXd duplicate_line(model.cols(), train_input.cols());
				auto&& reference = duplicate_line << train_input.row(train_index);
				for (size_t line = 0; line < model.cols() - 1; line++)
				{
					reference, train_input.row(train_index);
				}
				Eigen::MatrixXd update = learning_rate.array() * (1.0 / train_input.rows() * hx_sub_yx.row(train_index) * duplicate_line).transpose().array();
				//std::cout << duplicate_line << "\n\n";
				//std::cout << update << "\n\n";
				model -= update;
				std::cout << model << "\n\n";
			}
		}
	}

	//  T   -1    T
	//(X *X)   * X  * y
	//\note This matrix must be invertible, otherwise the result is undefined.
	Eigen::MatrixXd NormalEquation(const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
	{
		if ((train_input.transpose() * train_input).fullPivLu().isInvertible())
		{
			return (train_input.transpose() * train_input).inverse() * train_input.transpose() * train_output;
		}
		else
		{
			std::cout << "Not invertible!" << std::endl;
			return {};
		}
	}
}

int main_()
{
	std::cout << std::boolalpha;

	auto [train_input, train_output] = LinearRegression::RandomGenterateTrainSet(100);
	//std::cout << train_input << std::endl;
	//std::cout << train_output << std::endl;

	Eigen::MatrixXd model = Eigen::Vector2d(1, 200);
	Eigen::Vector2d learning_rate(0.005, 1);
	double limit = 0.05;
	size_t batch_size = 25;
	LinearRegression::GradientDescent(model, train_input, train_output, learning_rate, limit/*batch_size*/);
	std::cout << LinearRegression::NormalEquation(train_input, train_output) << std::endl;
	return 0;
}