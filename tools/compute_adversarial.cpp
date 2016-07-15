// #include STL headers
#include <string>

// #include support headers
#include "gflags/gflags.h" // maybe it needs to change "" with <>
#include "glog/logging.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include caffe files
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/caffe.hpp"

// using declarations
using caffe::Net;
using caffe::Caffe;
using cv::Mat;
using cv::imread;
using std::string;

// FLAGS definitions
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "The weights of the model to initialize the network.");
DEFINE_string(images, "",
    "The path to the folder that contains the image(s)");

// function definitions
void compute_perturbation(const Net<float>* network, string data) {

  LOG(INFO) << "Start compute_perturbation";

  Mat image;
  image = imread(data, CV_LOAD_IMAGE_COLOR);

  CHECK(!image.data) << "Could not open or find the image";

  LOG(INFO) << "Call DeepFool algorithm";
  // to be implemented
  //deepfool(network, image);

  LOG(INFO) << "DeepFool completed";

}

// main tool for computing adversarial images
int main(int argc, char** argv) {

  /* --- initialization --- */
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Computes an adversarial data given a dataset and "
      "a network (with its weights)\n"
      "Usage:\n"
      "   compute_adversarial [FLAGS] [NETWORK]\n"
      "   the network model needs to contain an image data layer.");

  // take the command line flags under the FLAGS_* name and remove them,
  // leaving only the other arguments
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Just to check the number of the arguments: " << argc;
  /* --- Check whether the number of argc is the required one --- */
  if (argc > 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_adversarial");
    return 1;
  }

  /* --- read images --- */
  CHECK_GT(FLAGS_images.size(), 0) << "Need a dataset to compute the "
      << "adversarial perturbations";
  std::string path_to_images = FLAGS_images;
  LOG(INFO) << "The image files are contained in the " << path_to_images
      << " folder";

  //
  LOG(INFO) << "Use CPU.";
  Caffe::set_mode(Caffe::CPU);

  /* --- network definition --- */
  // create and initialize the network which will be used
  // to create the adversarial examples
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model defition to compute "
      << "the gradients for the adversarial perturbations";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need models weights to compute "
      << "the gradients for the adversarial perturbations";
  LOG(INFO) << "Network defining...";

  FLAGS_logtostderr = 0; // disable log to stderr to have a cleaner output

  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  FLAGS_logtostderr = 1;

  LOG(INFO) << "Network initialized.";

  /* --- compute perturbation --- */
  // call compute_perturbation(images,network)
  compute_perturbation(&caffe_net, path_to_images);

  return 0;
}
