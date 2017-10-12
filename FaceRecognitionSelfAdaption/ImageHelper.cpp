#include "ImageHelper.h"



ImageHelper::ImageHelper()
{
}


ImageHelper::~ImageHelper()
{
}


void ImageHelper::read_images(const string& filename, vector<cv::Mat>& images, vector<int>& labels, cv::Size scale, char separator) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			cout << "Reading" + path + "; label:" + classlabel.c_str() << endl;
			cv::Mat image = cv::imread(path, 0);
			cv::Mat res;
			resize(image, res, scale, cv::INTER_CUBIC);
			images.push_back(res);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
