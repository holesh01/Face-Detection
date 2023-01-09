#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // Load the cascade classifier for detecting faces
    CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");

    // Load the input image
    Mat image = imread("image.jpg");

    // Convert the image to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Equalize the histogram
    Mat equalized;
    equalizeHist(gray, equalized);

    // Detect faces
    vector<Rect> faces;
    faceCascade.detectMultiScale(equalized, faces, 1.1, 3, 0, Size(20, 20));

    // Draw a rectangle around each face
    for (size_t i = 0; i < faces.size(); i++)
    {
        rectangle(image, faces[i], Scalar(255, 0, 0), 2);
    }

    // Show the image
    imshow("Faces", image);
    waitKey(0);

    return 0;
}
