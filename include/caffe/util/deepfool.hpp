#ifndef CAFFE_DEEPFOOL_HPP_
#define CAFFE_DEEPFOOL_HPP_

#include <string>

#include <caffe/util/classifier.hpp>

namespace caffe {
class DeepFool {
 public:

    // TODO: Provide more constructors
    DeepFool(const std::string& model_file,
             const std::string& trained_file,
             const std::string& mean_file,
             const std::string& label_file,
            // const size_t batch_size = 128,
             const int MAX_ITER = 50, const int Q = 2,
             const float OVERSHOOT = 0.02);

    /**
     *  @brief Compute the adversarial images for a image in OpenCV's
     *         Mat format.
     *
     */
    void adversarial(cv::Mat image, bool PREPROCESSING=false);

    /**
     *  @brief Compute the adversarial images for a vector that contains
     *         images in OpenCV's Mat format.
     */
    void adversarial(std::vector<cv::Mat> images, bool PREPROCESSING=false);

    /**
     *  @brief Preprocess a single image. WARNING: The object needs to have
     *         a classifier specified before this function is used.
     *
     *  TODO: Test if the classifier has been initialized
     */
    void Preprocess(cv::Mat& image) { classifier_.Preprocess(image); }


    /**
     *  @brief Preprocess a vector of images. WARNING: The object needs
     *         to have a classifier specified before this function is used.
     *
     *  TODO: Test if the classifier has been initialized
     */
    void Preprocess(std::vector<cv::Mat>& images) {
      classifier_.Preprocess(images);
    }


    inline int get_max_iterations() const { return max_iterations_; }
    inline int get_Q() const { return Q_; }
    inline int get_overshoot() const { return overshoot_; }

 private:
    Classifier classifier_;
    int max_iterations_;
    int Q_;
    float overshoot_;
    size_t number_of_labels_;

    // batch_size_ defines the maximum number of images that will
    // whose perturbations will be computed simultaneously to avoid
    // memory problems; used only when the input is a set of images
    size_t batch_size_;

};

} // namespace caffe

#endif // CAFFE_DEEPFOOL_HPP_
