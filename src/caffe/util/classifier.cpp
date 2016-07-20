/**
 *
 *  This file contains the definition of the Classifier class that is used by
 *  the /examples/
 *  The implementation of the Classifier has been moved here as it might be
 *  usefull to some one else. Additional functionality will be implemented
 *  as needed.
 *
 */

// STL libraries
#include <algorithm>
#include <string>

// caffe headers
#include "caffe/util/classifier.hpp"

// using declarations
using std::string;

namespace caffe {

// constructor of the classifier given a network, its weights,
// the mean image (used for preprocessing any new input) and
// a file containing the set of labels of the dataset
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  // Load the network that defines the Classifier
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  // check the number of channels - the classifier works with
  // Grayscale and RGB images
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";

  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  // Load the mean file used for preprocessing the data
  SetMean(mean_file);

  // Load the labels file which contains the labels used to train
  // (and test) the network and the classifier
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  // check if the output layer of the classifier has the right
  // amount of outputs
  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

// return the indiced of the top N values of vactor v
static std::vector<int> Argmax(const std::vector<float>& v, int N) {

  // make pairs of the predictions with indices
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i) {
    pairs.push_back(std::make_pair(v[i], i));
  }
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  // return the top N indices
  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);

  return result;
}

// method that classifies the image img and returns a vector with the
// top-N labels and their probability
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {

  // return a vector with the probibilites predicted for each class
  std::vector<float> output = Predict(img);

  // check that the user did not asked for more labels than
  // the amount provided by the dataset
  N = std::min<int>(labels_.size(), N);
  // return the indexes of the N labels with the max probabilities
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

// returns a vector of the probabilities of the predicted class labels
std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  // Copy the output layer (the predictions) to a std::vector
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}


/* UNCHANGED CODE */
/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::Preprocess(cv::Mat& img) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  img = sample_normalized;
}

std::vector<string> Classifier::get_layer_names() const {

  vector<string> layer_names = net_->layer_names();

  // TODO: Maybe remove the prinnting
  for (size_t i = 0; i < layer_names.size(); ++i) {
    std::cout << layer_names[i] << std::endl;
  }

  return layer_names;
}

std::vector<float> Classifier::InputGradientofClassifier(const cv::Mat& img,
                                                         int k) {


  Blob<float>* input_layer_1 = net_->input_blobs()[0];
  input_layer_1->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  std::vector<cv::Mat> input_channels;

  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  CHECK(k < labels_.size()) << "The classifier can discriminate "
      << "over fewer classes";

  Blob<float>* output_layer = net_->output_blobs()[0];
  float* output_diff = output_layer->mutable_cpu_diff();


  for (int i = 0; i < output_layer->count(); ++i) {
    output_diff[i] = 0;
  }
  output_diff[k] = 1;

  /* The following does the same thing as the last 4 lines
  // NOTE: Maybe use blob->count as a constructor initializer
  // NOTE: It might be better to intialize an array but vectors
  //       should act like arrays if we pass them as &vec[0]
  vector<float> output_val(labels_.size(), 0);
  output_val[k] = 1;

  CHECK(output_layer->count() == output_val.size())
      << "The number of diff to set is different from the number that "
      << "the output supports";

  caffe_copy(output_layer->count(), &output_val[0], output_diff);
  */

  net_->Backward();

  Blob<float>* input_layer = net_->input_blobs()[0];
  const float* begin = input_layer->cpu_diff();
  const float* end = begin + input_layer->count();
  return std::vector<float>(begin, end);
}

} // namespace caffe
