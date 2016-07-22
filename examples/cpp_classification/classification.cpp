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

// function that reads the UINT8 image_file image and converts it to
// a FLOAT32 image
cv::Mat read_image(std::string image_file) {

  // read image
  cv::Mat tmp_img = cv::imread(image_file, -1);
  CHECK(!tmp_img.empty()) << "Unable to decode image " << image_file;

  // convert a UINT8 image to a FLOAT32 image
  cv::Mat input_img;
  tmp_img.convertTo(input_img, CV_32FC1);

  return input_img;

}

// function that reads a set of images that are contained in the
// dir_name directory and returns a vector of the images
std::vector<cv::Mat> read_images_from_dir(std::string dir_name) {

  std::vector<cv::Mat> image_vec;

  /**
   * Simple solution; the glob() function is missing from the current
   * opencv library. Use the boost::filesystem (which might be heavy?).
   *
   * TODO: Update OpenCV so that it contains the cv::glob() function.
   *
  std::vector<std::string> image_file_names;
  cv::glob(dir_name, image_file_names);

  for (size_t i = 0; i < image_file_names.size(); ++i) {

    image_vec.push_back(read_image(image_file_names[i]));

  }
   */

  boost::filesystem::path p(dir_name);

  std::vector<boost::filesystem::path> tmp_vec;

  std::copy(boost::filesystem::directory_iterator(p),
            boost::filesystem::directory_iterator(),
            back_inserter(tmp_vec));

  std::vector<boost::filesystem::path>::const_iterator it = tmp_vec.begin();

  for (; it != tmp_vec.end(); ++it) {

    if (is_regular_file(*it)) {
      std::cout << it->string() << std::endl;
      image_vec.push_back(read_image(it->string()));
    }

  }

  return image_vec;
}

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
  std::vector<Classifier::Prediction> predictions;

  if (file_or_dir[file_or_dir.size() - 1] == '/') {
    std::cout << "----- Predicting for images contained in "
              <<  file_or_dir << " direcory -----" << std::endl;

    std::vector<cv::Mat> input_img = read_images_from_dir(file_or_dir);

    classifier.Preprocess(input_img);
    predictions = classifier.Classify(input_img, 5);

  } else if (string(file_or_dir.end() - 4, file_or_dir.end())
             == string(".jpg") ||
             string(file_or_dir.end() - 5, file_or_dir.end())
             == string(".JPEG")) {

    std::cout << "----- Predicting for the image " << file_or_dir
              << " -----" << std::endl;

    cv::Mat input_img = read_image(file_or_dir);

    classifier.Preprocess(input_img);
    predictions = classifier.Classify(input_img, 5);

  } else {
    LOG(ERROR) << "Images not found; check that you specified an image file "
              << "name or a directory path that containes images.";
  }


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
  std::cout << "Mat min and max " << min << " " << max << std::endl;
  std::cout << "Vector size: " << std::setprecision(8) << grads.size() << std::endl;
  std::cout << "Max value: " << *(std::max_element(grads.begin(), grads.end())) << std::endl;
  std::cout << "Min value: " << *(std::min_element(grads.begin(), grads.end())) << std::endl;
*/

}

#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
