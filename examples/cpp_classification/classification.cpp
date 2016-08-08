#include <caffe/caffe.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <boost/filesystem.hpp>

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
              << " mean.binaryproto labels.txt"
              << " img.jpg/directory_containing_images" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file_or_dir = argv[5];

  CHECK(!file_or_dir.empty()) << "No image or a directory containing images "
      << "was given.";

  // vector to get the predictions from the classifier
  std::vector<std::vector<Classifier::Prediction> > predictions;

  std::vector<std::string> image_names;

  // TODO: To be REMOVED; Just for checking
  std::vector<std::vector<float> > grads;

  if (file_or_dir[file_or_dir.size() - 1] == '/') {
    std::cout << "----- Predicting for images contained in "
              <<  file_or_dir << " direcory -----" << std::endl;

    std::vector<cv::Mat> input_img = read_images_from_dir(file_or_dir);

    classifier.Preprocess(input_img);
    predictions = classifier.Classify(input_img, 5);

    grads = classifier.InputGradientofClassifier(input_img, 0);

    image_names = read_names_from_dir(file_or_dir);

  } else if (string(file_or_dir.end() - 4, file_or_dir.end())
             == string(".jpg") ||
             string(file_or_dir.end() - 5, file_or_dir.end())
             == string(".JPEG")) {

    std::cout << "----- Predicting for the image " << file_or_dir
              << " -----" << std::endl;

    cv::Mat input_img = read_image(file_or_dir);

    classifier.Preprocess(input_img);
    predictions = classifier.Classify(input_img, 5);

    grads = classifier.InputGradientofClassifier(input_img, 0);

    image_names.push_back(file_or_dir);

  } else {
    LOG(ERROR) << "Images not found; check that you specified an image file "
              << "name or a directory path that containes images.";
  }

  // Print the top N predictions.
  for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << "Output for: " << image_names[i] << " image." << std::endl;

    for (size_t j = 0; j < predictions[0].size(); ++j) {
      Classifier::Prediction p = predictions[i][j];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
    }
  }

  return 0;
}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
