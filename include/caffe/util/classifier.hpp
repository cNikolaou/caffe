#ifndef CAFFE_CLASSIFIER_HPP_
#define CAFFE_CLASSIFIER_HPP_

// STL libraries
#include <string>
#include <vector>
#include <utility> // for pair
#include <memory>  // for shared_ptr but using boost's implementation

// other libraries
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <boost/shared_ptr.hpp>

// caffe header
#include <caffe/caffe.hpp>

namespace caffe {

typedef std::pair<std::string, float> Prediction;


class Classifier {
 public:
  Classifier(const std::string& model_file,
             const std::string& trained_file,
             const std::string& mean_file,
             const std::string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  boost::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<std::string> labels_;

  // method to set the mean_ variable
  void SetMean(const std::string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  // use the img to make a prediction rearding its class
  std::vector<float> Predict(const cv::Mat& img);

  // preprocess the image
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
};

} // namespace Caffe

#endif // CAFFE_CLASSIFIER_HPP_
