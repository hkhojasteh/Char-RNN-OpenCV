//Made with love by Hadi - Copyright(c) by Hadi Abdi Khojasteh - 2017-2018. All right reserved. / Email: hkhojasteh@iasbs.ac.ir, info@hadiabdikhojasteh.ir / Website: iasbs.ac.ir/~hkhojasteh, hadiabdikhojasteh.ir

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
typedef tuple<double, Mat1d, Mat1d, Mat1d, Mat1d, Mat1d, Mat1d> lossSet;
struct IncGenerator {
	uint32_t current_;
	IncGenerator(uint32_t start) : current_(start) {}
	uint32_t operator() () { return current_++; }
};

enumerate findWord(vector<enumerate>, char);
void updateAdagrad(Mat1d *, Mat1d *, Mat1d *);
vector<uint32_t> sample(Mat1d *, uint32_t, uint32_t);
lossSet lossFun(vector<enumerate>, vector<enumerate>, Mat1d);

uint32_t data_size, vocab_size;
Mat1d Wxh, Whh, Why, bh, by;							//model parameters

//hyperparameters
uint32_t hidden_size = 100;							//size of hidden layer of neurons
uint32_t seq_length = 5;							//number of steps to unroll the RNN for
double learning_rate = 1e-1;

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
		if (inchar[0] == 0) {
			continue;
		}
		data.push_back(inchar[0]);

		auto it = find_if(chars.begin(), chars.end(),
			[&](const char element) { return element == inchar[0]; });
		//If this is not in the char set
		if (it == end(chars)) {
			chars.push_back(inchar[0]);
			charenum.push_back(make_tuple(inchar[0], i));
		}
		i++;
	}
	fclose(inputfile);

	data_size = data.size();
	vocab_size = chars.size();
	printf("data has %d characters, %d unique.\n", data_size, vocab_size);
	vector<enumerate> char_to_ix = charenum;
	//reverse(charenum.begin(), charenum.end());
	//vector<enumerate> ix_to_char = charenum;

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
	double smooth_loss = -log(1.0 / vocab_size) * seq_length, loss = 0.0;

	Mat1d hprev;
	vector<enumerate> inputs, targets;
	for (uint32_t i = 0; i < 1e10; i++) {
		//Prepare inputs (we're sweeping from left to right in steps seq_length long)
		if (p + seq_length + 1 >= data.size() || n == 0) {
			hprev = Mat::zeros(hidden_size, 1, CV_32F);	//reset RNN memory
			p = 0;										//go from start of data
		}

		inputs.clear();
		targets.clear();
		for (uint32_t i = 0; i < seq_length && p + i + 1 < data.size() - 1; i++) {
			inputs.push_back(findWord(char_to_ix, data[p + i]));
			targets.push_back(findWord(char_to_ix, data[p + 1 + i]));
		}

		//Sample from the model now and then
		if (n % 100 == 0) {
			vector<uint32_t> sampWords = sample(&hprev, p + i, 200);
			for (uint32_t i = 0; i < sampWords.size(); i++) {
				printf("%c", get<0>(char_to_ix[sampWords[i]]));
			}
			printf("\n");
		}
		
		lossSet d = lossFun(inputs, targets, hprev);
		loss = get<0>(d);
		Mat1d dWxh, dWhh, dWhy, dbh, dby;
		dWxh = get<1>(d);
		dWhh = get<2>(d);
		dWhy = get<3>(d);
		dbh = get<4>(d);
		dby = get<5>(d);
		hprev = get<6>(d);

		smooth_loss = smooth_loss * 0.999 + loss * 0.001;
		if (n % 100 == 0) {
			printf("\niter %d, loss: %f\n", n, smooth_loss);
		}
		/*Wxh = dWxh;
		Whh = dWhh;
		Why = dWhy;

		bh = dbh;
		by = dby;*/

		//perform parameter update with Adagrad
		updateAdagrad(&Wxh, &dWxh, &mWxh);
		updateAdagrad(&Whh, &dWhh, &mWhh);
		updateAdagrad(&Why, &dWhy, &mWhy);
		updateAdagrad(&bh, &dbh, &mbh);
		updateAdagrad(&by, &dby, &mby);

		p += seq_length;								//move data pointer
		n += 1;											//iteration counter
	}
	return 0;
}

enumerate findWord(vector<enumerate> char_to_ix, char ichar) {
	uint32_t index = -1;
	auto it = find_if(char_to_ix.begin(), char_to_ix.end(),
		[&](const enumerate element) { return get<0>(element) == ichar; });
	if (it != end(char_to_ix)) {
		index = get<1>(*it);
	}
	return make_tuple(ichar, index);
}

void updateAdagrad(Mat1d * param, Mat1d * dparam, Mat1d * mem) {
	Mat1d powdparam;
	pow((*dparam), 2, powdparam);
	(*mem) += powdparam;
	(*param) += -learning_rate * (*dparam);
}

vector<uint32_t> sample(Mat1d * h, uint32_t seed_ix, uint32_t n) {
	//sample a sequence of integers from the model h is memory state, seed_ix is seed letter for first time step
	//h is changed in calling this function beacuse of hprev in main
	Mat1d x = Mat::zeros(vocab_size, 1, CV_32F);
	x[seed_ix][0] = 1.0;
	vector<uint32_t> ixes; 
	for (uint32_t i = 0; i < n; i++) {
		Mat1d t = (Wxh * x) + (Whh * (*h)) + bh;
		(*h) = Mat::zeros(t.size(), t.type());
		for (uint32_t i = 0; i < t.rows; i++) {
			(*h)[i][0] = tanh(t[i][0]);
		}
		Mat1d y = (Why * (*h)) + by;

		Mat1d expy;
		exp(y, expy);
		Mat1d p = expy / sum(expy)[0];
		p = p.reshape(1, 1);

		//Generates a random sample from a given 1-D array
		/*default_random_engine generator;
		discrete_distribution<int> distribution(p.begin(), p.end());
		vector<double> indices(p.size().width);
		generate(indices.begin(), indices.end(), [&generator, &distribution]() { return distribution(generator); });
		vector<int> incNumbers(p.size().width);
		IncGenerator gi(0);
		generate(incNumbers.begin(), incNumbers.end(), gi);*/
		x = Mat::zeros(vocab_size, 1, CV_32F);
		int randSelect = (uint32_t)rand() % vocab_size;
		x[randSelect][0] = 1.0;
		ixes.push_back(randSelect);
	}
	return ixes;
}

lossSet lossFun(vector<enumerate> inputs, vector<enumerate> targets, Mat1d hprev) {
	//inputs, targets are both list of integers.
	//     hprev is Hx1 array of initial hidden state
	//     returns the loss, gradients on model parameters, and last hidden state
	Mat1d xs = Mat::zeros(inputs.size(), vocab_size, CV_32F);
	Mat1d hs = Mat::zeros(inputs.size(), hprev.rows, hprev.type());
	Mat1d ys = Mat::zeros(vocab_size, 0, hprev.type());

	double loss = 0.0;
	Mat1d ps = Mat::zeros(Why.rows, 0, Why.type());;
	//forward pass
	for (uint32_t t = 0; t < inputs.size(); t++) {
		//encode in 1-of-k
		xs[t][get<1>(inputs[t])] = 1;

		//calculate hidden state matrix (hidden x vocab_size)
		Mat1d val;
		if (t == 0) {
			val = (Wxh * xs.row(t).t()) + (Whh * hprev.col(0)) + bh;
		}else{
			val = (Wxh * xs.row(t).t()) + (Whh * hs.row(t - 1).t()) + bh;
		}
		for (uint32_t i = 0; i < val.rows; i++) {
			hs[t][i] = tanh(val[i][0]);
		}
		Mat1d ysTemp = (Why * hs.row(t).t()) + by[t][0];				//unnormalized log probabilities for next chars
		hconcat(ys, ysTemp, ys);
		
		Mat1d expys;
		exp(ys.col(t), expys);											//probabilities for next chars
		hconcat(ps, expys / sum(expys)[0], ps);

		Mat1d logps;
		log(ps, logps);
		for (int32_t i = get<1>(targets[t]) - 1; i >= 0; i--) {
			loss += logps[t][i];										//softmax (cross-entropy loss)
		}
	}
	//backward pass: compute gradients going backwards
	Mat1d dWxh = Mat::zeros(Wxh.size(), Wxh.type());
	Mat1d dWhh = Mat::zeros(Whh.size(), Whh.type());
	Mat1d dWhy = Mat::zeros(Why.size(), Why.type());
	Mat1d dbh = Mat::zeros(bh.size(), bh.type());
	Mat1d dby = Mat::zeros(by.size(), by.type());
	Mat1d dhnext = Mat::zeros(hs.cols, 1, hs.type());
	for (int32_t t = inputs.size() - 1; t >= 0; t--){
		//compute derivative of error w.r.t the output probabilites - dE/dy[j] = y[j] - t[j]
		Mat1d dy = ps.col(t);
		dy[get<1>(targets[t])][0] -= 1;						//backprop into y
		dWhy += dy * hs.row(t);
		dby += dy;
		Mat1d dh = (Why.t() * dy) + dhnext;					//backprop into h
		
		Mat1d powhs;
		pow(hs.row(t), 2, powhs);							//backprop through tanh nonlinearity
		powhs = Mat::ones(powhs.size(), powhs.type()) - powhs;
		Mat1d dhraw = powhs.t().mul(dh);
		dbh += dhraw;
		dWxh += dhraw * xs.row(t);
		if (t == 0) {
			dWhh += dhraw * hprev.col(0).t();
		}else{
			dWhh += dhraw * hs.row(t - 1);
		}
		dhnext = Whh.t() * dhraw;
	}
	//clip to mitigate exploding gradients
	Mat1d tdWxh, tdWhh, tdWhy, tdbh, tdby;
	threshold(dWxh, tdWxh, 5.0, 5, THRESH_TRUNC);
	threshold(dWxh, tdWxh, -5.0, -5, THRESH_BINARY_INV);
	threshold(dWhh, tdWhh, 5.0, 5, THRESH_TRUNC);
	threshold(dWhh, tdWhh, -5.0, -5, THRESH_BINARY_INV);
	threshold(dWhy, tdWhy, 5.0, 5, THRESH_TRUNC);
	threshold(dWhy, tdWhy, -5.0, -5, THRESH_BINARY_INV);
	threshold(dbh, tdbh, 5.0, 5, THRESH_TRUNC);
	threshold(dbh, tdbh, -5.0, -5, THRESH_BINARY_INV);
	threshold(dby, tdby, 5.0, 5, THRESH_TRUNC);
	threshold(dby, tdby, -5.0, -5, THRESH_BINARY_INV);

	return make_tuple(loss, tdWxh, tdWhh, tdWhy, tdbh, tdby, hs.row(inputs.size() - 1).t());
}
