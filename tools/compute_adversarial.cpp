// #include STL headers
#include <string>
#include <vector>
#include <numeric> // for inner_product and accumulate
#include <cmath>   // for pow()
// TODO: Remove the iostream? Used only for verification?
#include <iostream>

// #include support headers
#include "gflags/gflags.h" // maybe it needs to change "" with <>
#include "glog/logging.h"

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

// #include caffe files
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/classifier.hpp"

// using declarations
using caffe::Net;
using caffe::Caffe;
using caffe::Blob;
using caffe::Classifier;
using std::string;
using std::vector;

// FLAGS definitions
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "The weights of the model to initialize the network.");
DEFINE_string(mean_file, "",
    "The path to the mean image that was used for preprocessing the data.");
DEFINE_string(labels_file, "",
    "The path to the file containing the labels that the network predicts.");
DEFINE_string(images, "",
    "The path to the folder that contains the image(s).");

// TODO: 1) Implement a DeepFool class that holds all the relevant information:
//          the classifier and the input and all the other
//       2) Implement all the optional arguments

// TODO: Remove this function
void print_img_dim(const cv::Mat& image) {

  std::cout << image.dims << " dimensions and " << std::endl
          << image.size() << std::endl << " and "
          << image.channels() << " channels." << std::endl;

}

// TODO: FInd a better way? use transformation() ??? Use caffes linear algebra??
vector<float> subvec(const vector<float>& first,
                     const vector<float>& second) {

  CHECK(first.size() == second.size()) <<
    "(deepfool) vectors are not of the same size";
/*
  std::cout << "first size: " << first.size() << "   second size: "
            << second.size() << std::endl;
*/
  vector<float> result(first.size(),0);

  for (size_t i = 0; i < first.size(); ++i) {
    result[i] = first[i] - second[i];
  }

  return result;
}

// TODO: naming; not actually the norm; move sqrt here?
float vector_norm(const vector<float>& v) {

  return inner_product(v.begin(), v.end(), v.begin(), 0.0);

}

// TODO: better naming
int ArgMin(const vector<float>& f_p, const vector<float>& w_pn) {

  CHECK(f_p.size() == w_pn.size()) << "(ArgMin) different vector dimensions";
  vector<float> tmp;
  for (size_t i = 0; i < f_p.size(); ++i) {
    tmp.push_back(abs(f_p[i]) / w_pn[i]);
  }

  vector<float>::const_iterator min_value = min_element(tmp.begin(), tmp.end());

  CHECK(min_value != tmp.end()) << "(ArgMin) no minimum value; should not occur";

  int ret = min_value - tmp.begin(); //size_t typedef

  return ret;

}


// TODO:
// number_of_labels - define
// l_hat - define
// ypologismos - define
// return value???
// w_prime - declaration and definition
// f_prime - declaration and definition

void deepfool(Classifier& classifier, const cv::Mat image) {

  cv::Mat x = image;

  // get the top prediction and the classifier which predictied it
  std::pair<string, float> top_label_prob = classifier.Classify(image)[0];
  std::vector<string> labels = classifier.get_labels();

  // unique labels se we find the one

  vector<string>::const_iterator labl = find(labels.begin(), labels.end(),
                                             top_label_prob.first);

  CHECK(labl != labels.end()) << "(deepfool function) The label was not found";

  // find the top classifier
  int top_classifier_index = labl - labels.begin();

  int i = 0;
  std::vector<std::vector<float> > r;
  // vector<> true_grad = classifier.gradient();

  cv::Mat temp_x = x;

  int MAX_ITER = 1;
  // TODO: Implement the parameters
  while (top_label_prob.first == classifier.Classify(x)[0].first &&
          i < MAX_ITER) {
    vector<float> top_classifier_grad =
            classifier.InputGradientofClassifier(x, top_classifier_index);

    vector<vector<float> > w_prime(labels.size() - 1,
                                   vector<float>(top_classifier_grad.size()));
    vector<float> w_prime_norm;


    vector<float> f_prime;
    vector<float> predictions = classifier.Predict(x); // get the prediction vec

    // TODO: optimize the for loop by removing the true_label from the list;
    //       change the loop to use one variable? Maybe more complex CODE
    int j = 0;
    for (int k = 0; k < labels.size(); ++k) {
      // if the label is different from the true_label
      if (labels[k] != top_label_prob.first) {
        vector<float> k_classifier_grad = classifier.InputGradientofClassifier(x, k);

        CHECK(k_classifier_grad.size() == top_classifier_grad.size()) <<
            "The " << k << "th classifier and the top classifier are not of "
            << "the same size";

        w_prime[j] = subvec(k_classifier_grad, top_classifier_grad);

        f_prime.push_back(predictions[k] - top_label_prob.second);
/*
        std::cout << k << " th classifier grad size "
                  << k_classifier_grad.size() << std::endl;
        std::cout << "w_prime size: " << w_prime.size() << " x "
                  << w_prime[k].size() << std::endl;
        std::cout << "f_prime size: " << f_prime.size() << std::endl;
*/
        w_prime_norm.push_back(std::sqrt(vector_norm(w_prime[j])));
        ++j;
      }

    }
/*
    std::cout << "w_prime size: " << w_prime.size() << " x "
              << w_prime[0].size() << std::endl;

    std::cout << "w_prime_norm size: " << w_prime_norm.size() << std::endl;

    std::cout << "f_prime size: " << f_prime.size() << std::endl;

    std::cout << "Printing f_prime: " << std::endl;
    for (int j = 0; j < f_prime.size(); ++j) {
      std::cout << j << " : " << f_prime[j] << std::endl;
    }
*/
/*
    std::cout << "Printing w_prime(1)" << std::endl;
    for (int j = 0; j < w_prime[0].size(); ++j) {
      std::cout << j << " : " << w_prime[0][j] << std::endl;
    }*/

    int l_hat = ArgMin(f_prime, w_prime_norm);

    float tmp = abs(f_prime[l_hat]) / std::pow(w_prime_norm[l_hat],2);

    // TODO: Write this in a more compact form using transform() or
    //       use the linear algebra methods from math_functions.cpp
    for (int j = 0; j < w_prime[l_hat].size(); ++j) {
      w_prime[l_hat][j] *= tmp;
    }

    r.push_back(w_prime[l_hat]);

    cv::Mat r_image(r[i], true);
    cv::Size input_size = classifier.get_geometry();

    std::cout << "Trying to create an image with size: " << input_size.height
              << " x " << input_size.width << std::endl;
    r_image = r_image.reshape(3, input_size.width);


    classifier.Preprocess(temp_x);

    std::cout << "r_image at iteration " << i << " has " << std::endl;
    print_img_dim(r_image);

    std::cout << "x at iteration " << i << " has " << std::endl;
    print_img_dim(x);

    std::cout << "temp_x at iteration " << i << " has " << std::endl;
    print_img_dim(temp_x);

    temp_x += r_image;

    // TODO: This is WRONG! The preprocessing routine should change
    x = temp_x;

    i++;
  }

  cv::imwrite( "temp_x.jpg", temp_x);
  //float r_hat = std::accumulate(r.begin(), r.end(), 0.0);
}


// TODO: Add documentation
void compute_perturbation(Classifier& classifier, string data) {

  // TODO: Maybe it needs to be removed - sanity checker
  LOG(INFO) << "Start compute_perturbation";

  // TODO: Improve functionality to support multiple images
  // load the image data
  cv::Mat image = cv::imread(data, CV_LOAD_IMAGE_COLOR);
  CHECK(!image.empty()) << "Could not open or find the image in " << data;

  // TODO: Remove the checker
  LOG(INFO) << "Call DeepFool algorithm";
  // to be implemented
  deepfool(classifier, image);

  // TODO: Remove the checker
  LOG(INFO) << "DeepFool completed";

}

// main tool for computing adversarial images
int main(int argc, char** argv) {

  // Initialization
  ::google::InitGoogleLogging(argv[0]);

  // TODO: Remove the extra information output?
  // FLAG testing
  FLAGS_logtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  // TODO: Make a more concrete message
  gflags::SetUsageMessage("Computes an adversarial data given a dataset and "
      "a network (with its weights)\n"
      "Usage:\n"
      "   compute_adversarial [FLAGS] [NETWORK] [WEIGHTS] [MEAN_FILE]"
      "[LABELS]\n"
      "   the network model needs to contain an image data layer.\n"
      "   -- The tool is under construction -- ");

  // take the command line flags under the FLAGS_* name and remove them,
  // leaving only the other arguments
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // TODO: Maybe needs removal - sanity check
  LOG(INFO) << "Just to check the number of the arguments: " << argc;

  // Check whether the number of arguments is the required one
  if (argc > 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_adversarial");
    return 1;
  }

  // Check the FLAGS provided to the tool
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition protocol buffer. "
      << "Use the --model flag to specify one.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need a trained model to initialize the"
      << "network. Use the --weights flag to specify a .caffemodel.";
  CHECK_GT(FLAGS_mean_file.size(), 0) << "Need a file containing the mean "
      << "image. Use the --mean_file flag to specify one.";
  CHECK_GT(FLAGS_labels_file.size(), 0) << "Need a file containing the labels.";
  CHECK_GT(FLAGS_images.size(), 0) << "Need a dataset to compute the "
      << "adversarial perturbations";

  // TODO: Maybe needs removal
  LOG(INFO) << "The image files are contained in the " << FLAGS_images
      << " folder";

  // TODO: Maybe the Caffe::CPU is not needed here.
  // It was here from the older code
  LOG(INFO) << "Use CPU.";
  Caffe::set_mode(Caffe::CPU);


  // Create and initialize the classifier which will be used
  // to create the adversarial examples
  LOG(INFO) << "Classifier defining...";

  FLAGS_logtostderr = 0; // disable log to stderr to have a cleaner output

  Classifier classifier(FLAGS_model, FLAGS_weights,
                        FLAGS_mean_file, FLAGS_labels_file);

  FLAGS_logtostderr = 1;

  // TODO: Sanity checker
  LOG(INFO) << "Classifier initialized.";

  // compute perturbation by callin the appropriate function
  // call compute_perturbation(images,network)
  compute_perturbation(classifier, FLAGS_images);

  return 0;
}
