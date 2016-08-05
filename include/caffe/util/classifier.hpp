#ifndef CAFFE_CLASSIFIER_HPP_
#define CAFFE_CLASSIFIER_HPP_

// STL libraries
#include <string>
#include <vector>
#include <utility> // for std::pair
#include <memory>  // for shared_ptr but using boost's implementation

// other libraries
// TODO: Use the same #ifdef at the functions
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <boost/shared_ptr.hpp>

// caffe header
#include <caffe/caffe.hpp>

namespace caffe {

class Classifier {
 public:

  // datatype that is returned from the Classifier after the classification
  typedef std::pair<std::string, float> Prediction;

  Classifier(const std::string& model_file,
             const std::string& trained_file,
             const std::string& mean_file,
             const std::string& label_file);

  /**
   * @brief Classify the images that are defined in the data blob (4D matrix)
   *        and return the top-N labels alongside their predictions (or the
   *        network's output, in general).
   *
   * TODO: IMPLEMENT IT -- maybe change the other Classify function
   */
  std::vector<std::vector<Prediction> >
  Classify(const Blob<float>* data, int N = 5);

  /**
   *  @brief Classify a cv::Mat object
   */
  std::vector<std::vector<Prediction> >
  Classify(const cv::Mat& img, int N = 5);

  /**
    * @brief Classify the image and return the top-N labels alongside their
    *        predictions (or the network's output, in general).
    *
    * TODO: Make it work with a vector<cv::Mat>
    */
  std::vector<std::vector<Prediction> >
  Classify(const std::vector<cv::Mat>& img, int N = 5);

  /**
   * @brief Get the network's output (predictions) given a blob that contains
   *        the data.
   */
  std::vector<float> Predict(const Blob<float>* data);

  /**
   * @brief Get the netwok's output (predictions) given the img image file;
   *        supports all the file types that are supported from the OpenCV
   *        library.
   *
   * TODO: Make it work with a vector<cv::Mat>
   */
  std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& img);

  std::vector<std::vector<float> > Predict(const cv::Mat& img);


  /**
   * @brief Get the gradient with respect to the input of the k-th classifier
   *
   * TODO: Add more detailed description
   */
  std::vector<std::vector<float> >
  InputGradientofClassifier(const std::vector<cv::Mat>& img, int k = 0);

  std::vector<std::vector<float> >
  InputGradientofClassifier(const cv::Mat& img, int k =0);

  /**
   * @brief Preprocess the given cv::Mat image. If UINT8_TO_FLOAT32 has
   *        value 'true' then the image will be transformed from a
   *        OpenCV::UINT8 to an OpenCV::FLOAT32 representation.
   *
   * TODO: Add similar functionality for Blob images?
   *
   */
  void Preprocess(cv::Mat& img, bool UINT8_TO_FLOAT32=false);

  /**
    * @brief Preprocess a vector of cv::Mat images. If UINT8_TO_FLOAT32 has
    *        value 'true' then the image will be transformed from a
    *        OpenCV::UINT8 to an OpenCV::FLOAT32 representation.
    */
  void Preprocess(std::vector<cv::Mat>& data, bool UINT8_TO_FLOAT32=false);

  /**
   * @brief Return the layer names of the defined network.
   */
  inline std::vector<string> get_layer_names() const {
    return net_->layer_names();
  }

  /**
   * @brief Returns the labels.
   */
  inline std::vector<string> get_labels() const { return labels_; }

  /**
   * @brief Returns the geometry of the input data.
   *
   * TODO: MAYBE improve the function
   */
  inline cv::Size get_geometry() const { return input_geometry_; }

  // TODO: Just a testing function -- REMOVE
  float print_mean_file(int i, int j) {
    return mean_.at<float>(i,j);
  }

  // TODO: Just a testing function -- REMOVE
  void save_mean_as_image() {
    cv::imwrite("mean_image.jpeg", mean_);
  }

  inline cv::Mat get_mean() const { return mean_; }

 private:
  // pointer for the underlaying network
  boost::shared_ptr<caffe::Net<float> > net_;
  // size of the input data
  std::vector<int> input_size_;
  // the labels that are related to the specific network
  std::vector<std::string> labels_;

  // TODO: Use USE_OPENCV
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;

  // method to set the mean_ variable used for preprocessing the cv::Mat images
  void SetMean(const std::string& mean_file);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  /**
   * Used for data preprocessing
   * TODO: Remove this function
   */
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void Input(const std::vector<cv::Mat>& data);
  void Input(const Blob<float>* data);

  void ImportData(const std::vector<cv::Mat>& data);
};

// related functions for reading images from the directory
cv::Mat read_image(std::string image_file);
std::vector<cv::Mat> read_images_from_dir(std::string dir_name);
std::vector<std::string> read_names_from_dir(std::string dir_name);

} // namespace Caffe

#endif // CAFFE_CLASSIFIER_HPP_
