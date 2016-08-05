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
             const int MAX_ITER = 50, const int Q = 2,
             const float OVERSHOOT = 0.02);

    // TODO: Provide overloaded adversarial functions
    void adversarial(cv::Mat image, bool PREPROCESSING=false);

    inline int get_max_iterations() const { return max_iterations_; }
    inline int get_overshoot() const { return overshoot_; }

 private:
    Classifier classifier_;
    int max_iterations_;
    int Q_;
    float overshoot_;
    size_t number_of_labels_;

};

} // namespace caffe

#endif // CAFFE_DEEPFOOL_HPP_
