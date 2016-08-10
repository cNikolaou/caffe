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

#include "caffe/blob.hpp"

// using declarations
using std::string;
using caffe::Blob;

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

  // check the number of channels of the given network -
  // the classifier works with Grayscale and RGB images
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";

  input_size_.push_back(input_layer->width());
  input_size_.push_back(input_layer->height());

  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  // Load the mean file used for preprocessing the data
  // when using cv::Mat data
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

// Compare two caffe::Prediction object
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}


// Get the indices of the top N values of vactor v
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
// top-N labels and their output (probibilites - if it is a softmax loss)
std::vector<std::vector<Classifier::Prediction> >
Classifier::Classify(const std::vector<cv::Mat>& img, int N) {

  // return a vector with the probibilites predicted for each class
  std::vector<std::vector<float> > output = Predict(img);

  // check that the user did not asked for more labels than
  // the amount provided by the dataset
  N = std::min<int>(labels_.size(), N);

  // return the indexes of the N labels with the max probabilities
  // TODO: Unwrap loop (by changing the Argmax function)?
  std::vector<std::vector<int> > maxN;
  //maxN.reserve(output.size());

  for (size_t i = 0; i < img.size(); ++i) {

    maxN.push_back(Argmax(output[i], N));

  }

  std::vector<std::vector<Prediction> > predictions;

  for (size_t j = 0; j < maxN.size(); ++j) {
    std::vector<Prediction> tmp_vec;
    for (int i = 0; i < N; ++i) {

      int idx = maxN[j][i];
      tmp_vec.push_back(std::make_pair(labels_[idx], output[j][idx]));

    }
    predictions.push_back(tmp_vec);
  }

  return predictions;
}

std::vector<std::vector<Classifier::Prediction> >
Classifier::Classify(const cv::Mat& img, int N) {

  // TODO: Do it in one line?
  std::vector<cv::Mat> img_mat;
  img_mat.push_back(img);

  return Classify(img_mat, N);

}


/*
// TODO: IMPLEMENT the function better -- not working as it is
void Classifier::Input(const Blob<float>* data) {

  vector<int> input_data_dim = data->shape();

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(input_data_dim[0], input_data_dim[1],
                       input_data_dim[2], input_data_dim[3]);

  net_->Reshape();

  float* input_data = input_layer.mutalbe_cpu_data();

}
*/
/*
// TODO: REMOVE the function -- Use ImportData()
void Classifier::Input(const std::vector<cv::Mat>& data) {

  // For C++98 compatibility instead of doing:
  // vector<int> vec = { ... }
  int tmp_dim[] = {data.size(), data[0].channels(), data[0].rows, data[0].cols};
  const vector<int> dimensions(tmp_dim, tmp_dim + 4);
  std::cout << "Input dimensions will be: " << dimensions[0] << " "
      << dimensions[1] << " " << dimensions[2] << " "
      << dimensions[3] << std::endl;
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(dimensions[0], dimensions[1],
                       dimensions[2], dimensions[3]);

  net_->Reshape();

  int K = data[0].channels();
  int H = data[0].rows;
  int W = data[0].cols;

  float* input_data = input_layer->mutable_cpu_data();


  for (size_t n = 0; n < dimensions[0]; ++n) {
    for (size_t k = 0; k < dimensions[1]; ++k) { //channel
      for (size_t h = 0; h < dimensions[2]; ++h) {
        for (size_t w = 0; w < dimensions[3]; ++w) {

          input_data[((n*K + k)*H + h)*W + w] =
                data[n].at<float>(h, w * 3 + k);

        }
      }
    }
  }
}
*/


// TODO: Make the function work with other image formats.
//       It works fine for CV_32FC3 image data. Test to find
//       out if it works for other data types as well.

// Import the images (data) to the input layer of the network
void Classifier::ImportData(const std::vector<cv::Mat>& data,
                            bool RESHAPE) {

  // For C++98 compatibility instead of doing:
  // vector<int> vec = { ... }
  // dimension: number of image data, number of channels, height, width
  int data_dimensions[]
            = { data.size(), data[0].channels(), data[0].rows, data[0].cols };

  std::cout << "Input dimensions will be: "
            << data_dimensions[0] << " " << data_dimensions[1] << " "
            << data_dimensions[2] << " " << data_dimensions[3] << std::endl;

  Blob<float>* input_layer = net_->input_blobs()[0];

  if (RESHAPE) {
    input_layer->Reshape(data_dimensions[0], data_dimensions[1],
                         data_dimensions[2], data_dimensions[3]);

    net_->Reshape();
  }

  float* input_data = input_layer->mutable_cpu_data();

  // TODO: Another alternative is to reshape the image matrx to a column one
  //       and then use matrix.col(0).copyTo(vec)
  //       Or use http://stackoverflow.com/questions/14303073/using-matati-j-in-opencv-for-a-2-d-mat-object

  int K = data_dimensions[1];
  int H = data_dimensions[2];
  int W = data_dimensions[3];

  for (size_t n = 0; n < data_dimensions[0]; ++n) {

    CHECK(data[n].type() == CV_32FC3)
        << "Image data should be in CV_32FC3 format to import it";

    for (size_t h = 0; h < H; ++h) {

      const float* img_row = data[n].ptr<float>(h);

      for (size_t w = 0; w < W; ++w) {

        for (size_t k = 0; k < K; ++k) {

//          input_data[((n*K + k)*H + h)*W + w] = img_row[w*K + k];
          input_data[n*K*W*H + k*H*W + h*W +w] = img_row[w*K + k];
        }
      }
    }
  }

  // TODO: REMOVE; Code used only for verification
  /*std::cout << " DATA : " << std::endl;

  for (size_t i = 0; i < input_layer->count(); i+=499) {
    std::cout << input_data[i] << "   ";
  }

  std::cout << std::endl << "IMG : " << std::endl;
  const float* img_row = data[0].ptr<float>(0);
  for (size_t i = 0; i < input_layer->count(); i+=499) {
    std::cout << img_row[i] << "   ";
  }*/
}


// get a vector of the probabilities of the predicted class
// labels for the images in the img vector
std::vector<std::vector<float> >
Classifier::Predict(const std::vector<cv::Mat>& img) {

  // TODO: Maybe
  // Reminder:
  // Preprocess before calling this function

  // Set the Blob of the input layer equal to the data in img
  //Input(img);  TODO: REMOVE this line
  ImportData(img);

  net_->Forward();

  std::vector<std::vector<float> > output;

  Blob<float>* output_layer = net_->output_blobs()[0];

  // TODO: REMOVE that code; Used only for verification
  //vector<int> output_shape = output_layer->shape();

  //std::cout << std::endl << "Channels: " << output_layer->channels() << std::endl;
  //std::cout << "Num: " << output_layer->num() << std::endl;
  int channels = output_layer->channels();
  const float* begin = output_layer->cpu_data();

  // Copy the output layer (the predictions) to a std::vector
  for (size_t n = 0; n < img.size(); ++n) {
    // copy the output of the n-th image to the return vector
    const float* start = begin + n*channels;
    const float* end = start + channels;
    std::vector<float> tmp(start, end);
    //std::cout << "Vector size: " << tmp.size() << std::endl;
    output.push_back(tmp);
  }

  return output;
}

// get a vector of the probabilities of the predicted class
// labels for the img image
vector<vector<float> >
Classifier::Predict(const cv::Mat& img) {

  std::vector<cv::Mat> vec;
  vec.push_back(img);

  return Predict(vec);
}

/*
// TODO: IMPLEMENT the function -- maybe it is not needed
std::vector<float> Classifier::Predict(const Blob<float>* data) {

  // Reminder:
  // Preprocess before calling this function
  Input(data);

  net_->Forward();

  // Copy the output layer (the predictions) to a std::vector
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin,end);
}
*/

// Load the mean file in binaryproto format.
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
 * layer.
 * TODO: Old code; not used - use the Input() instead; maybe REMOVE?
 */
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

// TODO: Old code; not used - use the other Preprocess() instead; maybe REMOVE?
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

// Preprocess the given image. This function is used before calling
// the classifier at the preprocessing step
void Classifier::Preprocess(cv::Mat& img) {

  // TODO: REMOVE unneccessary code that was used for
  //       verification (like the std::cout-s)

  //std::cout << img.at<float>(0,0) << " begin" << std::endl;

  // Convert the input image to the input image format of the network.
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else {
    std::cout << "Else" << std::endl;
    sample = img;
  }

  std::cout << (int) sample.at<uchar>(0,0) << " sample " << std::endl;
  // resize the image if neccessary
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_) {
    cv::resize(sample, sample_resized, input_geometry_);
    std::cout << "Resized" << std::endl;
    std::cout << "New size: " << sample_resized.rows << " x " << sample_resized.cols << std::endl;
  }
  else {
    sample_resized = sample;
    std::cout << "Not resized" << std::endl;
  }
  std::cout << (int) sample_resized.at<uchar>(0,0) << " sample_resized"<< std::endl;
  cv::Mat sample_float;

  std::cout << "Type output " << img.type() << std::endl;
  // transforme the image from UINT8 to FLOAT32 when needed
  if (img.type() == CV_8UC1 || img.type() == CV_8UC3) {

    if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

  } else if (img.type() == CV_32FC1 || img.type() == CV_32FC3) {

    LOG(INFO) << "ATTENTION: It is better to pass an image of CV_8UC1 or "
              << "CV_8UC3 type; otherwise you loose in prediction accuracy.";

    std::cout << "NOT CHANGE TO float32" << std::endl;
    sample_float = sample_resized;

  } else {

    LOG(ERROR) << "Please provite an image with type CV_8UC3, CV_8UC1 "
               << "CV_32FC1 or CV_32FC3";

  }

  // TODO: REMOVE the code; used only for verification
  std::cout << sample_float.at<float>(0,0) << " sample float" << std::endl;
  std::cout << mean_.at<float>(0,0) << " mean value B" << std::endl;
  std::cout << mean_.at<float>(0,1) << " mean value G" << std::endl;
  std::cout << mean_.at<float>(0,2) << " mean value R" << std::endl;
  std::cout << mean_.at<float>(0,3) << " mean value B" << std::endl;
  std::cout << mean_.at<float>(0,4) << " mean value G" << std::endl;
  std::cout << mean_.at<float>(0,5) << " mean value R" << std::endl;
  std::cout << mean_.at<float>(1,0) << " mean value B" << std::endl;
  std::cout << mean_.at<float>(1,1) << " mean value G" << std::endl;
  std::cout << mean_.at<float>(1,2) << " mean value R" << std::endl;

  // subtract the mean value
  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  img = sample_normalized;

  std::cout << img.at<float>(0,0) << " sample normalized"<< std::endl;
}


void Classifier::Preprocess(std::vector<cv::Mat>& data) {

  for (size_t i = 0; i < data.size(); ++i) {
    Preprocess(data[i]);
  }

}


std::vector<std::vector<float> >
Classifier::InputGradientofClassifier(const std::vector<cv::Mat>& img, int k) {


  // TODO: REMOVE Just for testing
  if (k%100 == 0) {
    std::cout << "First Grad; classifier " << k << std::endl;
  }


  Blob<float>* input_layer = net_->input_blobs()[0];

  // Check if the network has already been reshaped to represent
  // the data otherwise import the data and  perform a forward pass.
  // Instead of calling the Classify() or the Predict() function,
  // this process uses fewer resources as we don't need the predictions.
  if (input_layer->num() != img.size()) {

    LOG(INFO) << "Into the Forward pass of InputGradient with "
              << input_layer->num() << " defined and "
              << img.size() << " input images";

    // Import images to the network and perform a forward pass
    //Input(img);
    ImportData(img,false);
    net_->Forward();

  } // else the network has already been used for those images?
  // TODO: NOT ALWAYS THE CASE; find a better way to test it?

  CHECK(k < labels_.size()) << "The classifier can discriminate "
      << "over fewer classes";

  Blob<float>* output_layer = net_->output_blobs()[0];
  float* output_diff = output_layer->mutable_cpu_diff();

  // TODO: Maybe optimize to set all values to zero in one line?
  for (size_t i = 0; i < output_layer->count(); ++i) {
      output_diff[i] = 0;
  }

  /*LOG(INFO) << "Output is of size " << output_layer->channels()
            << "   " << output_layer->num()
            << "   " << output_layer->height()
            << "   " << output_layer->width();
*/
  int output_channels = output_layer->channels();

  for (size_t n = 0; n < img.size(); ++n) {
/*    LOG(INFO) << "Classifier " << k << " iteration " << n
              << " array position " << k + n*output_channels;*/
    output_diff[k + n*output_channels] = 1;
  }


  /*
  // The following does the same thing as the last 4 lines
  // (maybe older version)
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

  std::vector<std::vector<float> > input_grad;

  int image_size =
      input_layer->channels() * input_layer->height() * input_layer->width();

  const float* begin = input_layer->cpu_diff();
/*
  for (size_t n = 0; n < img.size(); ++n) {

    const float* start = begin + n*image_size;
    const float* end = start + image_size;
    std::vector<float> tmp(start, end);
    input_grad.push_back(tmp);

  }
*/

  size_t H = input_layer->shape(2);
  size_t W = input_layer->shape(3);

  for (size_t n = 0; n < input_layer->shape(0); ++n) {

    std::vector<float> tmp;
    const float* start = begin + n*image_size;

    for (size_t h = 0; h < H; ++h) {

      for (size_t w = 0; w < W; ++w) {

        for (size_t k = 0; k < input_layer->shape(1); ++k) {

          tmp.push_back(*(start + k*W*H + h*W + w));

        }
      }
    }

    input_grad.push_back(tmp);
  }

  return input_grad;
}

std::vector<std::vector<float> >
Classifier::InputGradientofClassifier(const cv::Mat& img, int k) {

  // TODO: Do it in one line?
  std::vector<cv::Mat> img_mat;
  img_mat.push_back(img);

  return InputGradientofClassifier(img_mat, k);
}

// related functions
// function that reads the UINT8 image_file image and converts it to
// a FLOAT32 image
cv::Mat read_image(std::string image_file) { //, bool UINT8_TO_FLOAT32) {

  // read image
  cv::Mat tmp_img = cv::imread(image_file, -1);
  CHECK(!tmp_img.empty()) << "Unable to decode image " << image_file;

  std::cout << " DATA IN: " << std::endl;
  std::cout << tmp_img.at<uchar>(0,0) << std::endl;

  cv::Mat input_img = tmp_img;

//  if (UINT8_TO_FLOAT32) {
    // convert a UINT8 image to a FLOAT32 image
//    tmp_img.convertTo(input_img, CV_32FC3);
//  }

  std::cout << " DATA TRAN: " << std::endl;
  std::cout << input_img.at<float>(0,0) << std::endl;

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
      //std::cout << it->string() << std::endl;
      image_vec.push_back(read_image(it->string()));
    }

  }

  return image_vec;
}


std::vector<std::string> read_names_from_dir(std::string dir_name) {

  std::vector<std::string> name_vec;

  boost::filesystem::path p(dir_name);

  std::vector<boost::filesystem::path> tmp_vec;

  std::copy(boost::filesystem::directory_iterator(p),
            boost::filesystem::directory_iterator(),
            back_inserter(tmp_vec));

  std::vector<boost::filesystem::path>::const_iterator it = tmp_vec.begin();

  for (; it != tmp_vec.end(); ++it) {

    if (is_regular_file(*it)) {
      //std::cout << it->string() << std::endl;
      name_vec.push_back(it->filename().string());
    }

  }

  return name_vec;
}

size_t Classifier::get_label_index(const std::string label_name) {

  std::vector<std::string>::iterator pos =
                              find(labels_.begin(), labels_.end(), label_name);

  CHECK(pos != labels_.end()) << "The label given is not contained in the "
                              << "labels recongized by the classifier";

  return pos - labels_.begin();
}

} // namespace caffe
