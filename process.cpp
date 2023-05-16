#include "process.h"

cv::Mat thresholdImage(const cv::Mat& inputImage) {
    cv::Mat grayImage, thresholdedImage;

    // Convert the input image to grayscale
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Apply simple thresholding
    const int thresholdValue = 127;  // You can adjust this value as needed
    const int maxValue = 255;
    cv::threshold(grayImage, thresholdedImage, thresholdValue, maxValue, cv::THRESH_BINARY_INV);

    return thresholdedImage;
}

cv::Mat cleanBinaryImage(const cv::Mat& binaryImage) {
    cv::Mat cleanedImage;

    // Define the structuring element
    int kernelSize = 3;  // You can adjust this as needed
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

    // Apply opening operation (erosion followed by dilation)
    cv::morphologyEx(binaryImage, cleanedImage, cv::MORPH_OPEN, kernel);

    return cleanedImage;
}

std::vector<cv::Rect> findRegions(const cv::Mat& cleanedImage, int minArea) {
    cv::Mat labels, stats, centroids;

    // Apply connected components with stats
    int numLabels = cv::connectedComponentsWithStats(cleanedImage, labels, stats, centroids);

    std::vector<cv::Rect> regions;

    // Loop over each label found
    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Rect rect(stats.at<int>(i, cv::CC_STAT_LEFT),
            stats.at<int>(i, cv::CC_STAT_TOP),
            stats.at<int>(i, cv::CC_STAT_WIDTH),
            stats.at<int>(i, cv::CC_STAT_HEIGHT));

        // Ignore small regions
        if (area > minArea) {
            regions.push_back(rect);
        }
    }

    return regions;
}

std::vector<RegionFeatures> computeFeatures(const cv::Mat& cleanedImage, const std::vector<cv::Rect>& regions) {
    std::vector<RegionFeatures> features;
    for (const cv::Rect& rect : regions) {
        cv::Mat regionImage = cleanedImage(rect);
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(regionImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            cv::Moments m = cv::moments(contour);
            cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);

            // Compute the oriented bounding box
            cv::RotatedRect orientedBoundingBox = cv::minAreaRect(contour);

            // Translate the oriented bounding box's center to the global space
            orientedBoundingBox.center += cv::Point2f(rect.tl());

            // Compute Hu Moments
            double hu[7];
            cv::HuMoments(m, hu);  // Now hu[0] through hu[6] contain your invariant Hu moments

            // Compute percentage filled
            double percentFilled = cv::countNonZero(regionImage) / (double)(rect.width * rect.height);

            // Compute scale invariant height-to-width ratio
            double hwRatio = (std::min)(orientedBoundingBox.size.width, orientedBoundingBox.size.height) / (std::max)(orientedBoundingBox.size.width, orientedBoundingBox.size.height);

            RegionFeatures feature;
            feature.centroid = centroid + cv::Point2f(rect.tl());
            std::copy(hu, hu + 7, feature.huMoments);
            feature.percentFilled = percentFilled;
            feature.hwRatio = hwRatio;
            feature.orientedBoundingBox = orientedBoundingBox;
            features.push_back(feature);
        }
    }
    return features;
}

double euclideanDistance(const RegionFeatures& a, const RegionFeatures& b) {
    double sum = 0.0;
    for (int i = 0; i < 7; i++) {
        sum += (a.huMoments[i] - b.huMoments[i]) * (a.huMoments[i] - b.huMoments[i]);
    }
    sum += (a.percentFilled - b.percentFilled) * (a.percentFilled - b.percentFilled);
    sum += (a.hwRatio - b.hwRatio) * (a.hwRatio - b.hwRatio);
    return std::sqrt(sum);
}

std::string nearestNeighbor(const RegionFeatures& instance, const std::vector<RegionFeatures>& trainingSet) {
    #ifdef max
    #undef max
    #endif
    double minDistance = std::numeric_limits<double>::max();
    std::string label;

    for (const RegionFeatures& feature : trainingSet) {
        double distance = euclideanDistance(instance, feature);
        if (distance < minDistance) {
            minDistance = distance;
            label = feature.label;
        }
    }
    return label;
}

std::vector<RegionFeatures> loadTrainingSet() {
    std::vector<RegionFeatures> trainingSet;
    std::ifstream file("training_set.csv");
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        RegionFeatures feature;
        std::string value;

        // Read centroid x and y
        std::getline(ss, value, ',');
        feature.centroid.x = std::stoi(value);
        std::getline(ss, value, ',');
        feature.centroid.y = std::stoi(value);

        for (int i = 0; i < 7; i++) {
            std::getline(ss, value, ',');
            feature.huMoments[i] = std::stod(value);
        }
        std::getline(ss, value, ',');
        feature.percentFilled = std::stod(value);

        std::getline(ss, value, ',');
        feature.hwRatio = std::stod(value);

        std::getline(ss, value, ',');
        feature.label = value;

        trainingSet.push_back(feature);
    }
    file.close();
    return trainingSet;
}
