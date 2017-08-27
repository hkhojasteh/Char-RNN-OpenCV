//This program created by love by Hadi - Copyright(c) by Hadi Abdi Khojasteh - Summer 2017. All right reserved. / Email: hkhojasteh@iasbs.ac.ir, info@hadiabdikhojasteh.ir / Website: iasbs.ac.ir/~hkhojasteh, hadiabdikhojasteh.ir

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef tuple<char, uint32_t> enumerate;
struct IncGenerator {
	uint32_t current_;
	IncGenerator(uint32_t start) : current_(start) {}
	uint32_t operator() () { return current_++; }
};

vector<uint32_t> sample(Mat1d, uint32_t, uint32_t);
void lossFun(vector<enumerate> inputs, vector<enumerate> targets, Mat1d hprev);

uint32_t data_size, vocab_size;
Mat1d Wxh, Whh, Why, bh, by;							//model parameters

uint32_t main() {
	srand(time(NULL));

	vector<char> data;
	vector<char> chars;
	vector<enumerate> charenum;
	FILE* inputfile;
	string fileName = "input.txt";
	inputfile = fopen(fileName.c_str(), "r");

	uint32_t i = 0;
	while (!feof(inputfile)) {
		char inchar[1] = { 0 };
		fscanf(inputfile, "%c", inchar);
		data.push_back(inchar[0]);

		auto it = find_if(chars.begin(), chars.end(),
			[&](const char element) { return element == inchar[0]; });
		//If this is not in the char set
		if (it == end(chars)) {
			chars.push_back(inchar[0]);
			charenum.push_back(make_tuple(inchar[0], i));
			i++;
		}
	}
	fclose(inputfile);

	data_size = data.size();
	vocab_size = chars.size();
	printf("data has %d characters, %d unique.\n", data_size, vocab_size);
	vector<enumerate> char_to_ix = charenum;
	reverse(charenum.begin(), charenum.end());
	vector<enumerate> ix_to_char = charenum;

	//hyperparameters
	uint32_t hidden_size = 100;							//size of hidden layer of neurons
	uint32_t seq_length = 25;							//number of steps to unroll the RNN for
	double learning_rate = 1e-1;

	Wxh.create(hidden_size, vocab_size);				//Or: Mat mat(2, 4, CV_64FC1);
	Whh.create(hidden_size, hidden_size);
	Why.create(vocab_size, hidden_size);
	double mean = 0.0, stddev = 1.0 / 3.0;				//99.7% of values will be inside [-1, +1] interval
	randn(Wxh, Scalar(mean), Scalar(stddev));			//input to hidden
	randn(Whh, Scalar(mean), Scalar(stddev));			//hidden to hidden
	randn(Why, Scalar(mean), Scalar(stddev));			//hidden to output
	bh = Mat::zeros(hidden_size, 1, CV_32F);			//hidden bias
	by = Mat::zeros(vocab_size, 1, CV_32F);				//output bias

	uint32_t n = 0, p = 0;
	//Make an array of zeros with the same shape and type as a Ws array.
	Mat1d mWxh = Mat::zeros(Wxh.size(), Wxh.type());
	Mat1d mWhh = Mat::zeros(Whh.size(), Whh.type());
	Mat1d mWhy = Mat::zeros(Why.size(), Why.type());
	Mat1d mbh = Mat::zeros(bh.size(), bh.type());		//memory variables for Adagrad
	Mat1d mby = Mat::zeros(by.size(), by.type());		//memory variables for Adagrad
	//loss at iteration 0
	double smooth_loss = -log(1.0 / vocab_size) * seq_length;

	Mat1d loss, dWxh, dWhh, dWhy, dbh, dby, hprev;
	vector<enumerate> inputs, targets;
	for (uint32_t i = 0; i < 1000; i++) {
		//Prepare inputs (we're sweeping from left to right in steps seq_length long)
		if (p + seq_length + 1 >= data.size() || n == 0) {
			hprev = Mat::zeros(hidden_size, 1, CV_32F);	//reset RNN memory
			p = 0;										//go from start of data
		}

		inputs.clear();
		targets.clear();
		for (uint32_t i = 0; i < seq_length && p + i < char_to_ix.size(); i++) {
			inputs.push_back(char_to_ix[p + i]);
			targets.push_back(char_to_ix[p + 1 + i]);
		}

		//Sample from the model now and then
		if (n % 100 == 0) {
			vector<uint32_t> sampWords = sample(hprev, p + i, 200);
			for (uint32_t i = 0; i < sampWords.size(); i++) {
				printf("%c", get<0>(ix_to_char[sampWords[i]]));
			}
			printf("\n");
		}
		
		lossFun(inputs, targets, hprev);

		p += seq_length;								//move data pointer
		n += 1;											//iteration counter
	}
	return 0;
}

vector<uint32_t> sample(Mat1d h, uint32_t seed_ix, uint32_t n) {
	//sample a sequence of integers from the model h is memory state,
	//     seed_ix is seed letter for first time step
	Mat1d x = Mat::zeros(vocab_size, 1, CV_32F);
	x[seed_ix][0] = 1.0;
	vector<uint32_t> ixes; 
	for (uint32_t i = 0; i < n; i++) {
		Mat1d t = (Wxh * x) + (Whh * h) + bh;
		Mat1d h = Mat::zeros(t.size(), t.type());
		for (uint32_t i = 0; i < t.rows; i++) {
			h[i][0] = tanh(t[i][0]);
		}
		Mat1d y = (Why * h) + by;
		Mat1d expy;
		exp(y, expy);
		Mat1d p = expy / sum(expy)[0];
		p = p.reshape(1, 1);

		//Generates a random sample from a given 1-D array
		default_random_engine generator;
		discrete_distribution<int> distribution(p.begin(), p.end());
		vector<double> indices(p.size().width);
		generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });
		vector<int> incNumbers(p.size().width);
		IncGenerator gi(0);
		generate(incNumbers.begin(), incNumbers.end(), gi);
		Mat1d x = Mat::zeros(vocab_size, 1, CV_32F);
		int randSelect = (uint32_t)rand() % vocab_size;
		x[randSelect][0] = 1.0;
		ixes.push_back(randSelect);
	}
	return ixes;
}

void lossFun(vector<enumerate> inputs, vector<enumerate> targets, Mat1d hprev) {
	//inputs, targets are both list of integers.
	//     hprev is Hx1 array of initial hidden state
	//     returns the loss, gradients on model parameters, and last hidden state
	Mat1d hs = hprev;
	double loss = 0.0;
	//forward pass
	for (uint32_t t = 0; t < inputs.size(); t++) {
		//encode in 1-of-k
		Mat1d xs = Mat::zeros(inputs.size(), vocab_size, CV_32F);
		xs[t][get<1>(inputs[t])] = 1;
		Mat1d val = (Wxh * xs.row(t).t());
		for (uint32_t i = 0; i < val.rows; i++) {
			hs[i][0] = tanh(val[i][0]);
			Mat1d temp = (Whh * hs[i - 1][0]);
			hs[i][0] += temp[i][0] + bh[i][0];				//hidden state
		}
		Mat1d ys = (Why * hs[t][0]) + by[t][0];				//unnormalized log probabilities for next chars
		//probabilities for next chars
		Mat1d ps = Mat::zeros(ys.size(), ys.type());
		double sum = 0.0;
		for (uint32_t i = 0; i < ys.rows; i++) {
			sum += ps[t][i];
			ps[t][i] = exp(ys[t][i]);
		}
		for (uint32_t i = 0; i < ys.rows; i++) {
			ps[t][0] = ps[t][0] / sum;
		}
		loss += -log(ps[t][get<1>(targets[t])]);			//softmax (cross-entropy loss)
	}
}
