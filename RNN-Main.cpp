//Made with love by Hadi - Copyright(c) by Hadi Abdi Khojasteh - 2017-2018. All right reserved. / Email: hkhojasteh [at] iasbs.ac.ir, info [at] hadiabdikhojasteh.ir / Website: iasbs.ac.ir/~hkhojasteh, hadiabdikhojasteh.ir

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

typedef tuple<double, Mat1d, Mat1d, Mat1d, Mat1d, Mat1d, Mat1d> lossSet;

enumerate findWord(vector<enumerate>, char);
void updateAdagrad(Mat1d *, Mat1d, Mat1d *);
vector<uint32_t> sample(Mat1d *, uint32_t, uint32_t);
uint32_t selectByDistribution(Mat1d);
lossSet lossFun(vector<enumerate>, vector<enumerate>, Mat1d);

uint32_t data_size, vocab_size;
Mat1d Wxh, Whh, Why, bh, by;							//model parameters

//hyperparameters
uint32_t hidden_size = 100;								//size of hidden layer of neurons
uint32_t seq_length = 10;								//number of steps to unroll the RNN for
double learning_rate = 1e-1;							//Learning rate is 0.1

uint32_t iterations = 1e10;								//number of iterations

class reader;
class RNN;
typedef tuple<char, uint32_t> enumerate;
typedef tuple<Mat1d, Mat1d, Mat1d, Mat1d, Mat1d> paramSet;

Mat1d initRandomMat(uint32_t rows, uint32_t cols) {
	Mat1d output;
	output.create(rows, cols);
	double mean = 0.0, stddev = 1.0 / 3.0;				//99.7% of values will be inside [-1, +1] interval
	randn(output, Scalar(mean), Scalar(stddev));		//Make random value matrix
	return output * 0.01;
}

class reader {
private:
	vector<char> data, chars;
	uint32_t data_size, vocab_size, p, seq_length;
	FILE* inputfile;
public:
	vector<enumerate> charenum, char_to_ix;
	reader(string path, uint32_t seq_length) {
		inputfile = fopen(path.c_str(), "r");

		uint32_t i = 0;
		while (!feof(inputfile)) {
			char inchar[1] = { 0 };
			fscanf(inputfile, "%c", inchar);
			if (inchar[0] == 0) {
				continue;
			}
			inchar[0] = (char)tolower((int)inchar[0]);
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
		for (uint32_t i = 0; i < vocab_size; i++) {
			printf("%c ", chars[i]);
		}
		printf("\n\n");
		char_to_ix = charenum;
		p = 0;
		this->seq_length = seq_length;
	}
	void nextBatch(vector<enumerate> * inputs, vector<enumerate> * targets) {
		uint32_t input_start = p;
		uint32_t input_end = p + seq_length;

		//Prepare inputs (we're sweeping from left to right in steps seq_length long)
		if (p + seq_length + 1 >= data.size()) {
			p = 0;		//go from start of data and reset pointer
		}

		inputs->clear();
		targets->clear();
		for (uint32_t i = 0; i < seq_length && p + i + 1 < data.size() - 1; i++) {
			inputs->push_back(findWord(char_to_ix, data[p + i]));
			targets->push_back(findWord(char_to_ix, data[p + 1 + i]));
		}
		p += seq_length;
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
	bool justStarted() {
		return p == 0;
	}
protected:
};

class RNN {
private:
	uint32_t hidden_size, vocab_size, seq_length;
	double learning_rate;
	Mat1d Wxh, Whh, Why, bh, by;
public:
	RNN(uint32_t hidden_size, uint32_t vocab_size, uint32_t seq_length, double learning_rate) {
		//Hyper parameters
		this->hidden_size = hidden_size;
		this->vocab_size = vocab_size;
		this->seq_length = seq_length;
		this->learning_rate = learning_rate;

		//Model parameters
		this->Wxh = initRandomMat(hidden_size, vocab_size);		//input to hidden - Mat mat(2, 4, CV_64FC1)
		this->Whh = initRandomMat(hidden_size, hidden_size);	//hidden to hidden
		this->Why = initRandomMat(vocab_size, hidden_size);		//hidden to output
		this->bh = Mat::zeros(hidden_size, 1, CV_32F);			//hidden bias
		this->by = Mat::zeros(vocab_size, 1, CV_32F);			//output bias

		/*

		//Memory vars for adagrad
		Mat1d mWxh = Mat::zeros(Wxh.size(), Wxh.type());
		Mat1d mWhh = Mat::zeros(Whh.size(), Whh.type());
		Mat1d mWhy = Mat::zeros(Why.size(), Why.type());
		Mat1d mbh = Mat::zeros(bh.size(), bh.type());			//memory variables for Adagrad
		Mat1d mby = Mat::zeros(by.size(), by.type());			//memory variables for Adagrad

		*/
	}
	void forward(vector<enumerate> inputs, Mat1d hprev, Mat1d * xs, Mat1d * hs, Mat1d * ps) {
		//inputs, targets are both list of integers.
		//     hprev is Hx1 array of initial hidden state
		//	   returns the next chars probabilities, model parameters, and last hidden state (with pointer as input)
		*xs = Mat::zeros(inputs.size(), vocab_size, CV_32F);		//one-hot inputs
		*hs = Mat::zeros(inputs.size(), hprev.rows, hprev.type());	//hidden states
		*ps = Mat::zeros(Why.rows, 0, Why.type());					//softmax probabilities
		Mat1d ys = Mat::zeros(vocab_size, 0, hprev.type());			//outputs

		//forward pass
		for (uint32_t t = 0; t < inputs.size(); t++) {
			//encode in 1-of-k (Convert to a one-hot vector)
			(*xs)[t][get<1>(inputs[t])] = 1;

			//calculate hidden state matrix (hidden x vocab_size)
			Mat1d val;
			if (t == 0) {
				val = (Wxh * (*xs).row(t).t()) + (Whh * hprev.col(0)) + bh;
			}else{
				val = (Wxh * (*xs).row(t).t()) + (Whh * (*hs).row(t - 1).t()) + bh;
			}
			for (uint32_t i = 0; i < val.rows; i++) {
				(*hs)[t][i] = tanh(val[i][0]);
			}
			Mat1d ysTemp = (Why * (*hs).row(t).t()) + by[t][0];		//unnormalized log probabilities for next chars
			hconcat(ys, ysTemp, ys);
		
			Mat1d expys;
			exp(ys.col(t), expys);
			hconcat((*ps), expys / sum(expys)[0], (*ps));			//probabilities for next chars
		}
	}
	paramSet backward(Mat1d xs, Mat1d hprev, Mat1d hs, Mat1d ps, vector<enumerate> targets) {
		//Compute gradients going backwards
		Mat1d dWxh = Mat::zeros(Wxh.size(), Wxh.type());
		Mat1d dWhh = Mat::zeros(Whh.size(), Whh.type());
		Mat1d dWhy = Mat::zeros(Why.size(), Why.type());
		Mat1d dbh = Mat::zeros(bh.size(), bh.type());
		Mat1d dby = Mat::zeros(by.size(), by.type());
		Mat1d dhnext = Mat::zeros(hs.cols, 1, hs.type());
		for (int32_t t = seq_length - 1; t >= 0; t--){
			//compute derivative of error w.r.t the output probabilites - dE/dy[j] = y[j] - t[j]
			Mat1d dy = ps.col(t);
			//backprop into y. The gradient of the cross-entropy loss is really as copying over the distribution and subtracting 1 from the correct class.
			dy[get<1>(targets[t])][0] -= 1;
			dWhy += dy * hs.row(t);
			dby += dy;
			Mat1d dh = (Why.t() * dy) + dhnext;		//backprop into h
		
			Mat1d powhs;
			pow(hs.row(t), 2, powhs);				//backprop through tanh nonlinearity
			powhs = Mat::ones(powhs.size(), powhs.type()) - powhs;
			Mat1d dhraw = powhs.t().mul(dh);
			dbh += dhraw;
			dWxh += dhraw * xs.row(t);
			//ToDo: Change this negative index by increasing start index of hs and remove hprev from here
			if (t == 0) {
				dWhh += dhraw * hprev.col(0).t();
			}else{
				dWhh += dhraw * hs.row(t - 1);
			}
			dhnext = Whh.t() * dhraw;
		}
		//clip to mitigate explodfing gradients
		Mat1d tdWxh, tdWhh, tdWhy, tdbh, tdby;
		tdWxh = clip(dWxh, -5.0, 0.5);
		tdWhh = clip(dWhh, -5.0, 0.5);
		tdWhy = clip(dWhy, -5.0, 0.5);
		tdbh = clip(dbh, -5.0, 0.5);
		tdby = clip(dby, -5.0, 0.5);

		return make_tuple(tdWxh, tdWhh, tdWhy, tdbh, tdby);
	}
	double modelLoss(Mat1d ps, vector<enumerate> targets) {
		double loss = 0.0;
		for (uint32_t t = 0; t < seq_length; t++) {
			for (int32_t i = get<1>(targets[t]) - 1; i >= 0; i--) {
				loss += -log(ps[t][i]);		//softmax (cross-entropy loss)
			}
		}
		return loss;
	}
	void updateModel(Mat1d dWxh, Mat1d dWhh, Mat1d dWhy, Mat1d dbh, Mat1d dby) {
		Mat1d mWxh = Mat::zeros(Wxh.size(), Wxh.type());
		Mat1d mWhh = Mat::zeros(Whh.size(), Whh.type());
		Mat1d mWhy = Mat::zeros(Why.size(), Why.type());
		Mat1d mbh = Mat::zeros(bh.size(), bh.type());		//memory variables for Adagrad
		Mat1d mby = Mat::zeros(by.size(), by.type());		//memory variables for Adagrad
		updateAdagrad(&Wxh, dWxh, &mWxh);
		updateAdagrad(&Whh, dWhh, &mWhh);
		updateAdagrad(&Why, dWhy, &mWhy);
		updateAdagrad(&bh, dbh, &mbh);
		updateAdagrad(&by, dby, &mby);
	}
	void updateAdagrad(Mat1d * param, Mat1d dparam, Mat1d * mem) {
		Mat1d powdparam, sqrtmem;
		pow(dparam, 2, powdparam);
		(*mem) += powdparam;
		sqrt((*mem), sqrtmem);
		(*param) += (-learning_rate * dparam) / (sqrtmem + 1e-8);
	}
	vector<uint32_t> sample(Mat1d * h, uint32_t seed_ix, uint32_t n) {
		//sample a sequence of integers from the model h is memory state, seed_ix is seed letter for first time step
		//h is changed in calling this function beacuse of hprev in main
		Mat1d x = Mat::zeros(vocab_size, 1, CV_32F);
		x[seed_ix][0] = 1.0;
		//Set up our one-hot encoded input vector based on the seed character.
		vector<uint32_t> ixes;
		for (uint32_t i = 0; i < n; i++) {
			Mat1d t = (Wxh * x) + (Whh * (*h)) + bh;
			(*h) = Mat::zeros(t.size(), t.type());
	#pragma omp parallel
			{
	#pragma omp for schedule(dynamic) ordered
				for (int i = 0; i < t.rows; i++) {
					(*h)[i][0] = tanh(t[i][0]);
				}
			}
			Mat1d y = (Why * (*h)) + by;

			Mat1d expy;
			exp(y, expy);
			Mat1d p = expy / sum(expy)[0];
			//Generates a random sample from a given 1-D array
			//int32_t randSelect = selectByDistribution(p);
			double min, max;
			Point min_loc, max_loc;
			cv::minMaxLoc(p, &min, &max, &min_loc, &max_loc);
			uint32_t randSelect = max_loc.y;

			x[randSelect][0] = 1.0;
			ixes.push_back(randSelect);
		}
		return ixes;
	}
	void train(reader data) {
		uint32_t iterNum = 0;
		double smooth_loss = -log(1.0 / vocab_size) * seq_length, loss = 0.0;
		Mat1d hprev;
		vector<enumerate> inputs, targets;
		Mat1d xs, hs, ps;
		for (uint32_t i = 0; i < iterations; i++) {
			if (data.justStarted()) {
				hprev = Mat::zeros(hidden_size, 1, CV_32F);	//reset RNN memory
			}
			data.nextBatch(&inputs, &targets);
			forward(inputs, hprev, &xs, &hs, &ps);
			paramSet o = backward(xs, hprev, hs, ps, targets);
			loss = modelLoss(ps, targets);
			updateModel(get<0>(o), get<1>(o), get<2>(o), get<3>(o), get<4>(o));
			smooth_loss = smooth_loss * 0.999 + loss * 0.001;
			hprev = hs.row(seq_length - 1).t();

			//Sample from the model now and then
			if (iterNum % 100 == 0) {
				vector<uint32_t> sampWords = sample(&hprev, get<1>(inputs[0]), 200);
				for (uint32_t i = 0; i < sampWords.size(); i++) {
					printf("%c", get<0>(data.char_to_ix[sampWords[i]]));
				}
				printf("\niter %d, loss: %f\n", iterNum, smooth_loss);
			}
			iterNum++;
		}
	}
protected:
};

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
		inchar[0] = (char)tolower((int)inchar[0]);
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
	for (uint32_t i = 0; i < vocab_size; i++) {
		printf("%c ", chars[i]);
	}
	printf("\n\n");
	vector<enumerate> char_to_ix = charenum;
	//reverse(charenum.begin(), charenum.end());
	//vector<enumerate> ix_to_char = charenum;

	Wxh = initRandomMat(hidden_size, vocab_size);		//input to hidden - Mat mat(2, 4, CV_64FC1)
	Whh = initRandomMat(hidden_size, hidden_size);		//hidden to hidden
	Why = initRandomMat(vocab_size, hidden_size);		//hidden to output
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
	for (uint32_t i = 0; i < iterations; i++) {
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
			vector<uint32_t> sampWords = sample(&hprev, p, 200);
			for (uint32_t i = 0; i < sampWords.size(); i++) {
				printf("%c", get<0>(char_to_ix[sampWords[i]]));
			}
			printf("\n");
		}
		
		lossSet resl = lossFun(inputs, targets, hprev);
		loss = get<0>(resl);
		Mat1d dWxh, dWhh, dWhy, dbh, dby;
		dWxh = get<1>(resl);
		dWhh = get<2>(resl);
		dWhy = get<3>(resl);
		dbh = get<4>(resl);
		dby = get<5>(resl);
		hprev = get<6>(resl);

		smooth_loss = smooth_loss * 0.999 + loss * 0.001;
		if (n % 100 == 0) {
			printf("\niter %d, loss: %f\n", n, smooth_loss);
		}
		//perform parameter update with Adagrad
		updateAdagrad(&Wxh, dWxh, &mWxh);
		updateAdagrad(&Whh, dWhh, &mWhh);
		updateAdagrad(&Why, dWhy, &mWhy);
		updateAdagrad(&bh, dbh, &mbh);
		updateAdagrad(&by, dby, &mby);

		/*Wxh = dWxh;
		Whh = dWhh;
		Why = dWhy;
		bh = dbh;
		by = dby;*/

		p += seq_length;								//move data pointer
		n += 1;											//iteration counter
	}
	scanf("%d", NULL);
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

void updateAdagrad(Mat1d * param, Mat1d dparam, Mat1d * mem) {
	Mat1d powdparam, sqrtmem;
	pow(dparam, 2, powdparam);
	(*mem) += powdparam;
	sqrt((*mem), sqrtmem);
	(*param) += (-learning_rate * dparam) / (sqrtmem + 1e-8);
}

vector<uint32_t> sample(Mat1d * h, uint32_t seed_ix, uint32_t n) {
	//sample a sequence of integers from the model h is memory state, seed_ix is seed letter for first time step
	//h is changed in calling this function beacuse of hprev in main
	Mat1d x = Mat::zeros(vocab_size, 1, CV_32F);
	x[seed_ix][0] = 1.0;
	//Set up our one-hot encoded input vector based on the seed character.
	vector<uint32_t> ixes;
	for (uint32_t i = 0; i < n; i++) {
		Mat1d t = (Wxh * x) + (Whh * (*h)) + bh;
		(*h) = Mat::zeros(t.size(), t.type());
#pragma omp parallel
		{
#pragma omp for schedule(dynamic) ordered
			for (int i = 0; i < t.rows; i++) {
				(*h)[i][0] = tanh(t[i][0]);
			}
		}
		Mat1d y = (Why * (*h)) + by;

		Mat1d expy;
		exp(y, expy);
		Mat1d p = expy / sum(expy)[0];
		//Generates a random sample from a given 1-D array
		//int32_t randSelect = selectByDistribution(p);
		double min, max;
		Point min_loc, max_loc;
		cv::minMaxLoc(p, &min, &max, &min_loc, &max_loc);
		uint32_t randSelect = max_loc.y;

		x[randSelect][0] = 1.0;
		ixes.push_back(randSelect);
	}
	return ixes;
}

uint32_t selectByDistribution(Mat1d p) {
	//distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x.
	vector<double> accumulatedProb(p.rows + 1);
	accumulatedProb[0] = p[0][0];
#pragma omp parallel
	{
#pragma omp for schedule(dynamic) ordered
		for (int i = 1; i < p.rows; i++)
			accumulatedProb[i] = accumulatedProb[i - 1] + p[i][0];
	}
	random_device rd;		//Will be used to obtain a seed for the random number engine
	mt19937 gen(rd());		//Standard mersenne_twister_engine seeded with rd()
	uniform_real_distribution<> dis(0.0, 1.0);
	double selectedItem = dis(gen);

	auto it = find_if(accumulatedProb.begin(), accumulatedProb.end(),
		[&](double element) { return element >= selectedItem; });
	if (it != end(accumulatedProb)) {
		uint32_t dis = distance(accumulatedProb.begin(), it);
		if (dis < accumulatedProb.size()){
			return dis;
		}else{
			return -1;
		}
	}else{
		return -1;
	}
}

lossSet lossFun(vector<enumerate> inputs, vector<enumerate> targets, Mat1d hprev) {
	//inputs, targets are both list of integers.
	//     hprev is Hx1 array of initial hidden state
	//     returns the loss, gradients on model parameters, and last hidden state
	Mat1d xs = Mat::zeros(inputs.size(), vocab_size, CV_32F);			//one-hot inputs
	Mat1d hs = Mat::zeros(inputs.size(), hprev.rows, hprev.type());		//hidden states
	Mat1d ys = Mat::zeros(vocab_size, 0, hprev.type());					//outputs
	Mat1d ps = Mat::zeros(Why.rows, 0, Why.type());						//softmax probabilities

	double loss = 0.0;
	//forward pass
	for (uint32_t t = 0; t < inputs.size(); t++) {
		//encode in 1-of-k (Convert to a one-hot vector)
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
		for (int32_t i = get<1>(targets[t]) - 1; i >= 0; i--) {
			loss += -log(ps[t][i]);										//softmax (cross-entropy loss)
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
		dy[get<1>(targets[t])][0] -= 1;						//backprop into y. The gradient of the cross-entropy loss is really as copying over the distribution and subtracting 1 from the correct class.
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
	//clip to mitigate explodfing gradients
	Mat1d tdWxh, tdWhh, tdWhy, tdbh, tdby;
	tdWxh = clip(dWxh, -5.0, 0.5);
	tdWhh = clip(dWhh, -5.0, 0.5);
	tdWhy = clip(dWhy, -5.0, 0.5);
	tdbh = clip(dbh, -5.0, 0.5);
	tdby = clip(dby, -5.0, 0.5);

	return make_tuple(loss, tdWxh, tdWhh, tdWhy, tdbh, tdby, hs.row(inputs.size() - 1).t());
}

Mat1d clip(Mat1d inMatrx, double min, double max){
	Mat1d thrtempf, thrtemps;
	threshold(inMatrx, thrtempf, max, max, THRESH_TRUNC);
	threshold(inMatrx, thrtemps, min, min, THRESH_BINARY_INV);
	return thrtempf + thrtemps;
}