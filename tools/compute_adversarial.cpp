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
#include "caffe/util/deepfool.hpp"

// using declarations
using caffe::DeepFool;
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


// function to test the output images
void save_image(cv::Mat img, std::string name, bool FLOAT32_TO_UINT8 = false) {

  cv::Mat tmp_img;

  if (FLOAT32_TO_UINT8) {
    tmp_img.convertTo(img, CV_8UC3);
  } else {
    tmp_img = img;
  }

  cv::imwrite(name, tmp_img);
  std::cout << name << " saved!" << std::endl << std::endl;

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

  // TODO: REMOVE this - sanity check
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

  // TODO: Maybe REMOVE this
  LOG(INFO) << "The image files are contained in the " << FLAGS_images
      << " folder";

  // TODO: Maybe the Caffe::CPU is not needed here.
  // It was here from the older code
  //LOG(INFO) << "Use CPU.";
  //Caffe::set_mode(Caffe::CPU);


  // Create and initialize the classifier which will be used
  // to create the adversarial examples
  LOG(INFO) << "Classifier defining...";

  FLAGS_logtostderr = 0; // disable log to stderr to have a cleaner output

/*
  Classifier classifier(FLAGS_model, FLAGS_weights,
                        FLAGS_mean_file, FLAGS_labels_file);
*/

  DeepFool df(FLAGS_model, FLAGS_weights, FLAGS_mean_file, FLAGS_labels_file);

  FLAGS_logtostderr = 1;

  // TODO: Sanity checker
  LOG(INFO) << "Classifier initialized.";

  // read image
  cv::Mat img = caffe::read_image(FLAGS_images);

  Classifier cl(FLAGS_model, FLAGS_weights, FLAGS_mean_file, FLAGS_labels_file);


  // TODO: REMOVE the code; it is used only for checking purposes

  double min, max;

  // TODO: Solve problem with reading and writting
  std::cout << "--- Start: --- " << std::endl;
  std::cout << "Cols x Rows before: " << img.rows << " x " << img.cols << std::endl;
  std::cout << "Vals: " << (int) img.at<uchar>(0,0) << " " << (int) img.at<uchar>(100,100) << std::endl;
  cv::minMaxLoc(img, &min, &max);
  std::cout << "Before permute: min and max " << min << " " << max << std::endl << std::endl;
  save_image(img, "Start_image.jpeg");

  cv::Mat im2;// = cv::Mat(img).reshape(0, img.cols);
  cv::transpose(img, im2);
  std::cout << "--- Permute: --- " << std::endl;
  std::cout << "Cols x Rows after: " << im2.rows << " x " << im2.cols << std::endl;
  std::cout << "Vals: " << (int) im2.at<unsigned char>(0,0) << " " << (int) im2.at<uchar>(100,100) << std::endl;
  cv::minMaxLoc(im2, &min, &max);
  std::cout << "Before preprocess (after permute): min and max " << min << " " << max << std::endl << std::endl;
  save_image(im2, "Permuted_image.jpeg");

  cl.Preprocess(img);
  std::cout << "--- Preprocess without permute: --- " << std::endl;
  std::cout << "Cols x Rows after pp wo permute: " << img.rows << " x " << img.cols << std::endl;
  std::cout << "Vals: " << img.at<float>(0,0) << " " << img.at<float>(100,100) << std::endl;
  cv::minMaxLoc(img, &min, &max);
  std::cout << "After preprocess: min and max " << min << " " << max << std::endl << std::endl;
  save_image(img, "PreprocessWO_image.jpeg");

  cl.Preprocess(im2);
  std::cout << "--- Preprocess with permute: --- " << std::endl;
  std::cout << "Cols x Rows after pp with permute: " << img.rows << " x " << img.cols << std::endl;
  std::cout << "vals: " << im2.at<float>(0,0) << " " << im2.at<float>(100,100) << std::endl;
  cv::minMaxLoc(im2, &min, &max);
  std::cout << "After preprocess with permute: min and max " << min << " " << max << std::endl << std::endl;
  save_image(im2, "PreprocessWI_image.jpeg");

  cv::minMaxLoc(im2, &min, &max);
  std::cout << "After preprocess: min and max " << min << " " << max << std::endl;
  std::cout << std::endl << std::endl;
  LOG(INFO) << "Call deepfool";

  std::cout << cl.print_mean_file(0,0) << std::endl;
  cl.save_mean_as_image();

  // compute perturbations
  df.adversarial(img);

  LOG(INFO) << "Adversarial Found";
  // compute perturbation by calling the appropriate function
  // call compute_perturbation(images,network)
  //compute_perturbation(classifier, FLAGS_images);

  return 0;
}
