#include <cmath>
#include <cstdlib>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

namespace LogisticRegression
{
	//Z(X₁,X₂)=1(θ₁X₁+θ₂X₂>800)
	//Z(X₁,X₂)=0(θ₁X₁+θ₂X₂<=800)
	//θ₁=10,θ₂=150,X₂=1
	std::tuple<Eigen::MatrixX2d, Eigen::VectorXd> RandomGenterateTrainSet(size_t counts)
	{
		Eigen::MatrixX2d train_input(counts, 2);
		Eigen::VectorXd train_output(counts);
		for (size_t i = 0; i < counts; i++)
		{
			double X₁ = std::rand() % 100 + 20, X₂ = 1;
			train_input(i, 0) = X₁;
			train_input(i, 1) = X₂;
			double θ₁ = 10, θ₂ = 150;
			double loss = /*std::rand() % 11 - 5;*/0;
			double hx = θ₁ * X₁ + θ₂ * X₂ + loss;
			train_output[i] = hx > 800 ? 1 : 0;
		}
		/*for (size_t i = 0; i < train_input.cols() - 1; i++)
		{
			auto&& col_i = train_input.col(i);
			col_i = (col_i.array() - col_i.sum() / train_input.rows() / 2) / (col_i.maxCoeff() - col_i.minCoeff());
		}*/
		return { train_input/*.normalized()*/, train_output };
	}

	//        -z
	//hx=1/1+e
	//          i      i      i         i
	//-1/m * ∑y log(hx )+(1-y )log(1-hx )
	double LossFunction(const Eigen::MatrixXd& model, const Eigen::MatrixXd& train_input, const Eigen::MatrixXd& train_output)
	{
		auto z = (train_input * model).array();
		//translation relative position
		auto length = z.maxCoeff() / 2 + z.minCoeff() / 2;
		auto z_translation = z - length;
		auto hx = 1 / (1 + Eigen::exp(-1 * z_translation));
		//(0,1)
		constexpr double multiple = 0.999999;
		auto hx_limit = hx * multiple;
		//std::cout << hx_limit << "\n\n";
		//std::cout << train_output << "\n\n";
		return -1.0 / train_output.rows() * (train_output.array() * Eigen::log(hx_limit) + (1 - train_output.array()) * Eigen::log(1 - hx_limit)).sum();
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
				auto z = (train_input * model).array();
				//translation relative position
				auto length = z.maxCoeff() / 2 + z.minCoeff() / 2;
				auto z_translation = z - length;
				auto hx = 1 / (1 + Eigen::exp(-1 * z_translation));
				//(0,1)
				constexpr double multiple = 0.999999;
				auto hx_limit = hx * multiple;
				auto hx_sub_yx = hx_limit.matrix() - train_output;
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
				std::cout << LossFunction(model, train_input, train_output) << "\n\n";
				model -= update;
				//std::cout << model << "\n\n";
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
	Eigen::Vector2d learning_rate(0.0001, 0.001);
	double limit = 0.05;
	size_t batch_size = 100;
	//std::cout << LogisticRegression::LossFunction(model, train_input, train_output) << std::endl;
	LogisticRegression::GradientDescent(model, train_input, train_output, learning_rate, batch_size);
	return 0;
}