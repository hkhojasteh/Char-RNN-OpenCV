//This program created by love by Hadi - Copyright(c) by Hadi Abdi Khojasteh - Summer 2017. All right reserved. / Email: hkhojasteh@iasbs.ac.ir, info@hadiabdikhojasteh.ir / Website: iasbs.ac.ir/~hkhojasteh, hadiabdikhojasteh.ir

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main() {
	vector<char> data;
	vector<char> chars;
	FILE* inputfile;
	string fileName = "input.txt";
	inputfile = fopen(fileName.c_str(), "r");

	while (!feof(inputfile)) {
		char inchar[1] = { 0 };
		fscanf(inputfile, "%c", inchar);
		data.push_back(inchar[0]);

		auto it = find_if(chars.begin(), chars.end(),
			[&](const char element) { return element == inchar[0]; });
		//If this is not in the char set
		if (it == end(chars)) {
			chars.push_back(inchar[0]);
		}
	}

	uint32_t data_size = data.size();
	uint32_t vocab_size = chars.size();
	printf("data has %d characters, %d unique.\n", data_size, vocab_size);

	//hyperparameters
	uint32_t hidden_size = 100;					//size of hidden layer of neurons
	uint32_t seq_length = 25;					//number of steps to unroll the RNN for
	uint32_t learning_rate = 1e-1;

	//model parameters
	Mat1d Wxh(hidden_size, vocab_size);			// Or: Mat mat(2, 4, CV_64FC1);
	Mat1d Whh(hidden_size, vocab_size);
	Mat1d Why(hidden_size, vocab_size);
	double mean = 0.0, stddev = 1.0 / 3.0;		// 99.7% of values will be inside [-1, +1] interval

	randn(Wxh, Scalar(mean), Scalar(stddev));	//input to hidden
	randn(Whh, Scalar(mean), Scalar(stddev));	//hidden to hidden
	randn(Why, Scalar(mean), Scalar(stddev));	//hidden to output
	Mat bh = Mat::zeros(1, hidden_size, CV_32F);//hidden bias
	Mat by = Mat::zeros(1, vocab_size, CV_32F); //output bias

	uint32_t n, p = 0;
	//Make an array of zeros with the same shape and type as a Ws array.
	Mat mWxh = Mat::zeros(Wxh.size(), Wxh.type());
	Mat mWhh = Mat::zeros(Whh.size(), Whh.type());
	Mat mWhy = Mat::zeros(Why.size(), Why.type());
	Mat mbh = Mat::zeros(bh.size(), bh.type());	//memory variables for Adagrad
	Mat mby = Mat::zeros(by.size(), by.type());	//memory variables for Adagrad


	return 0;
}