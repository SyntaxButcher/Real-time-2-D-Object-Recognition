#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "filter.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>

using namespace cv;
using namespace std;

struct RegionFeatures {
    cv::Point2f centroid; // Translation invariant
    double huMoments[7];  // Translation, rotation, and scale invariant
    double percentFilled; // Translation, rotation, and scale invariant
    double hwRatio;       // Translation and rotation invariant, can be made scale invariant
    cv::RotatedRect orientedBoundingBox;
    std::string label; // Label of the object
};

cv::Mat thresholdImage(const cv::Mat& inputImage);
cv::Mat cleanBinaryImage(const cv::Mat& binaryImage);
std::vector<cv::Rect> findRegions(const cv::Mat& cleanedImage, int minArea = 500);
std::vector<RegionFeatures> computeFeatures(const cv::Mat& cleanedImage, const std::vector<cv::Rect>& regions);
double euclideanDistance(const RegionFeatures& a, const RegionFeatures& b);
std::string nearestNeighbor(const RegionFeatures& instance, const std::vector<RegionFeatures>& trainingSet);
std::vector<RegionFeatures> loadTrainingSet();