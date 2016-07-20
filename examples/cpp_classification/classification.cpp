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

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img, 5);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }

  std::cout << "Check the gradient of the classifier" << std::endl;
  std::vector<float> grads = classifier.InputGradientofClassifier(img, 0);
  for (int k = 0; k < grads.size(); ++k) {
    std::cout << k << " " << grads[k] << std::endl;
  }
/*
  std::cout << "M = " << std::endl << " " << img << std::endl << std::endl;
  double min, max;
  cv::minMaxLoc(img, &min, &max);
  std::cout << "Mat min and max " << min << " " << max << std::endl;*/
  std::cout << "Vector size: " << std::setprecision(8) << grads.size() << std::endl;
  std::cout << "Max value: " << *(std::max_element(grads.begin(), grads.end())) << std::endl;
  std::cout << "Min value: " << *(std::min_element(grads.begin(), grads.end())) << std::endl;


}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
