#pragma once
#include<opencv2\opencv.hpp>
#include<opencv2\face.hpp>
#include<iostream> 
using namespace std;

class ImageHelper
{
public:
	ImageHelper();
	~ImageHelper();
	void static read_images(const string& filename, vector<cv::Mat>& images, vector<int>& labels, cv::Size scale, char separator= ';');
};

