
#include "cv_puttextzh.h"

int main() {
    cv::Mat image = cv::imread("colorhouse.png");
    cv::putTextZH(
        image,
        "Hello, OpenCV����",
        cv::Point(10, image.rows / 2),
        CV_RGB(255, 255, 255),
        20
    );
    cv::imshow("ԭͼ", image);
    cv::waitKey(0);

    return 0;
}
