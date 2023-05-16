#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "process.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;

    // Load the training set from the CSV file
    std::vector<RegionFeatures> trainingSet = loadTrainingSet();

    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        cv::imshow("Video", frame);

        // Thresholding
        cv::Mat binaryImage = thresholdImage(frame);

        // Cleaning
        cv::Mat cleanedImage = cleanBinaryImage(binaryImage);

        // Find regions
        std::vector<cv::Rect> regions = findRegions(cleanedImage);

        // Compute features
        std::vector<RegionFeatures> regionFeatures = computeFeatures(cleanedImage, regions);

        // Classify each region
        for (RegionFeatures& feature : regionFeatures) {
            feature.label = nearestNeighbor(feature, trainingSet);
            std::cout << "Predicted label: " << feature.label << std::endl;
            // Display the predicted label next to the region's centroid
            cv::putText(frame, feature.label, feature.centroid + cv::Point2f(20, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
        }

        // Draw regions and features on the original frame
        for (const RegionFeatures& feature : regionFeatures) {
            cv::circle(frame, feature.centroid, 5, cv::Scalar(0, 0, 255), -1);

            // Draw the oriented bounding box
            cv::Point2f vertices[4];
            feature.orientedBoundingBox.points(vertices);
            for (int i = 0; i < 4; i++)
                cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

            // Draw the percentage filled and height/width ratio
            std::string text = "Filled: " + std::to_string(feature.percentFilled) + ", HW Ratio: " + std::to_string(feature.hwRatio);
            // cv::putText(frame, text, feature.centroid, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }

        // Display the original frame with regions, features, and labels
        cv::imshow("Regions, Features, and Labels", frame);


        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        if (key == 'N' || key == 'n') {
            // Enter training mode
            std::string label;
            std::cout << "Enter label for the current object: ";
            std::cin >> label;

            // Find the region with the largest area
            RegionFeatures* largestRegion = &regionFeatures[0];
            for (RegionFeatures& feature : regionFeatures) {
                if (feature.orientedBoundingBox.size.area() > largestRegion->orientedBoundingBox.size.area()) {
                    largestRegion = &feature;
                }
            }

            // Add the largest region's features to the training set
            largestRegion->label = label;
            trainingSet.push_back(*largestRegion);

            // After the loop, save the training set to a CSV file
            std::ifstream infile("training_set.csv");
            bool writeHeader = !infile.good();
            infile.close();

            std::ofstream file("training_set.csv", std::ios::app); // Open in append mode

            RegionFeatures& feature = trainingSet.back(); // Get the last entry
            file << feature.centroid.x << "," << feature.centroid.y << ","
                << feature.huMoments[0] << "," << feature.huMoments[1] << "," << feature.huMoments[2] << "," << feature.huMoments[3] << ","
                << feature.huMoments[4] << "," << feature.huMoments[5] << "," << feature.huMoments[6] << ","
                << feature.percentFilled << "," << feature.hwRatio << ","
                << feature.label << "\n";
            file.close();
        }
        else if (key == 'q') {
            break;
        }

    }

    delete capdev;
    return(0);
}