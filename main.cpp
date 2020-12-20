#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem> // C++17 https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
#include <algorithm>
#include <random>
#include <time.h>

typedef std::vector<std::vector<float>> Matrix;


const std::string base = std::filesystem::current_path().generic_string(); // BASE PROJECT PATH

/* TRAIN PATH */
const std::string train_cannon_path = base + "/train/cannon";
const std::string train_cellphone_path = base + "/train/cellphone";

/* TEST PATH */
const std::string test_cannon_path = base + "/test/cannon/image_0021.jpg";
const std::string test_cellphone_path = base + "/test/cellphone/image_0001.jpg";
static int cannon_size = 0;
static int total_size = 0;

double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivate_sigmoid(const double y) {
    return sigmoid(y) * (1 - sigmoid(y));   
}

float dot(const std::vector<float>& v1, const std::vector<float>& v2) {
	if (v1.size() != v2.size())
		throw;
	float sum = 0.0f;
	for (int i = 0; i < v1.size(); i++) {
		sum += (v1[i] * v2[i]);
	}
	return sum;
}

void push_horizontal(Matrix& matrix, const std::vector<float>& v) {
	if (!matrix.empty() && matrix.size() != v.size())
		throw;
	for (int i = 0; i < matrix.size(); i++) {
		matrix[i].push_back(v[i]);
	}
}

template<typename T>
void shuffle(std::vector<T>& v) {
	auto rng = std::default_random_engine{};
	std::shuffle(v.begin(), v.end(), rng);
}

void calculate_weights(std::vector<float>& weights, const float lr, const float error, const float d_sigmoid, const std::vector<float>& input) {
    float coef = lr * error * d_sigmoid;
    for (int i = 0; i < input.size(); i++) {
        weights[i] += coef * input[i];
    }
}

void train(const Matrix& matrix, const std::vector<float>& target, std::vector<float>& weights, const float lr, const int iteration) {
    
    for (int i = 0; i < iteration; i++) {
        for (int j = 0; j < matrix.size(); j++) {
            float sum = dot(weights, matrix[j]);
            float d_sigmoid = derivate_sigmoid(sum);
            float error = target[j] - sigmoid(sum);
            
            int k = 0;
            for (std::vector<float>::iterator it = weights.begin(); it != weights.end(); ++it) {
                *it += (lr * error * d_sigmoid * matrix[j][k++]);
            }
            
        }
    }
}

float test(const char* path, std::vector<float>& weights) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    cv::Mat dest;
    cv::resize(image, dest, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
    dest = dest.reshape(1, 1);
    float max = *std::max_element(dest.begin<uchar>(), dest.end<uchar>());
    std::vector<float> v;
    
    for (auto i = dest.begin<uchar>(); i != dest.end<uchar>(); i++) {
        v.push_back((float)*i / max);
    }
    v.push_back(1); /* bias */
    return sigmoid(dot(v, weights));
}

int main() {

    clock_t tStart = clock();

    Matrix images;

    for (const auto& entry : std::filesystem::directory_iterator(train_cannon_path)) {
        cv::Mat input = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        cv::Mat dest;
        cv::resize(input, dest, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
        float max = *std::max_element(dest.begin<uchar>(), dest.end<uchar>());
        images.push_back(std::vector<float>());

        for (int i = 0; i < dest.rows; i++) {
            for (int j = 0; j < dest.cols; j++) {
                cv::Vec3b intensity = dest.at<cv::Vec3b>(i, j);
                images[cannon_size].push_back(intensity.val[0] / max); // blue
                images[cannon_size].push_back(intensity.val[1] / max); // green
                images[cannon_size].push_back(intensity.val[2] / max); // red
            }
        }
       
        cannon_size++;
    }

    total_size += cannon_size;
    for (const auto& entry : std::filesystem::directory_iterator(train_cellphone_path)) {
        cv::Mat input = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        cv::Mat dest;
        cv::resize(input, dest, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
        float max = *std::max_element(dest.begin<uchar>(), dest.end<uchar>());
        images.push_back(std::vector<float>());
            
        for (int i = 0; i < dest.rows; i++) {
            for (int j = 0; j < dest.cols; j++) {
                cv::Vec3b intensity = dest.at<cv::Vec3b>(i, j);
                images[total_size].push_back(intensity.val[0] / max); // blue
                images[total_size].push_back(intensity.val[1] / max); // red
                images[total_size].push_back(intensity.val[2] / max); // green
            }
        }

        total_size++;
    }

    
    /* BIAS */
    std::vector<float>* b = new std::vector<float>(100);
    std::fill(b->begin(), b->end(), 1);
    push_horizontal(images, *b);
    
    /* IMAGES SHUFFLE*/
    shuffle(images);

    /* WEIGHTS */
    std::vector<float>* w = new std::vector<float>(images[0].size());
    std::fill(w->begin(), w->end(), 0);

    /* TARGET */
    std::vector<float>* t = new std::vector<float>((total_size));
    std::fill(t->begin(), t->begin() + cannon_size, 0); // CANNON class 0
    std::fill(t->begin() + cannon_size, t->end(), 1); // CELLPHONE class 1

    /* TARGET SHUFFLE*/
    shuffle(*t);

    /* TRAIN PERCEPTRON */
    train(images, *t, *w, 0.001f, 1000);

    /* TEST */
    printf("CANON : %.8fs\n", test(test_cannon_path.c_str(), *w));
    printf("CELLPHONE : %.8fs\n", test(test_cellphone_path.c_str(), *w));
    printf("Time taken: %.2fs\n", (float)((clock() - tStart)) / CLOCKS_PER_SEC);
    
	std::cout << "Done.\n";
	return 0;
}