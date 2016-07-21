#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <iomanip>

#include <caffe/util/classifier.hpp>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file = argv[5];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat input_img = cv::imread(file, -1);
  //cv::Mat input_img =
  CHECK(!input_img.empty()) << "Unable to decode image " << file;

  cv::Mat img_float;
  input_img.convertTo(img_float, CV_32FC1);
/*
  std::vector<cv::Mat> img;
  img.push_back(input_img2);
  int tmp[] = {img.size(), img[0].channels(), img[0].rows, img[0].cols};
  const std::vector<int> dimensions(tmp, tmp + 4);

  std::cout << "The input image is: " << std::endl << img[0] << std::endl;

  // TODO: Another alternative is to reshape the image matrx to a column ont
  //       and then use matrix.col(0).copyTo(vec)
  //       Or use http://stackoverflow.com/questions/14303073/using-matati-j-in-opencv-for-a-2-d-mat-object
  int count = 0;
  for (int n = 0; n < dimensions[0]; ++n) {
    for (int c = 0; c < dimensions[1]; ++c) {
      std::cout << std::endl << "Channel: " << c << std::endl;
      for (int h = 0; h < dimensions[2]; ++h) {
        std::cout << std::endl << "Column: " << h << std::endl;
        for (int r = 0; r < dimensions[3]; ++r) {
          std::cout << img[n].at<float>(h,r*3 + c) << " ";
          count++;
        }
      }
    }
    std::cout << std::endl << "Counted " << count << " pixels" << std::endl;

  }*/
  /**/
  classifier.Preprocess(img_float);
  std::vector<Classifier::Prediction> predictions =
        classifier.Classify(img_float, 5);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Classifier::Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
/*
  std::cout << "Check the gradient of the classifier" << std::endl;
  std::vector<float> grads = classifier.InputGradientofClassifier(img, 0);
  for (int k = 0; k < grads.size(); ++k) {
    std::cout << k << " " << grads[k] << std::endl;
  }*/
/*
  std::cout << "M = " << std::endl << " " << img << std::endl << std::endl;
  double min, max;
  cv::minMaxLoc(img, &min, &max);
  std::cout << "Mat min and max " << min << " " << max << std::endl;*/
  //std::cout << "Vector size: " << std::setprecision(8) << grads.size() << std::endl;
  //std::cout << "Max value: " << *(std::max_element(grads.begin(), grads.end())) << std::endl;
  //std::cout << "Min value: " << *(std::min_element(grads.begin(), grads.end())) << std::endl;


}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
