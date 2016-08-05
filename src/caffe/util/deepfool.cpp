#include <string>
#include <numeric> // inner_product

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include <boost/lexical_cast.hpp>

#include <caffe/util/deepfool.hpp>
#include <caffe/util/classifier.hpp>


namespace caffe{

  DeepFool::DeepFool(const std::string& model_file,
                     const std::string& trained_file,
                     const std::string& mean_file,
                     const std::string& label_file,
                     const int MAX_ITER, const int Q,
                     const float OVERSHOOT)
              : classifier_(model_file, trained_file, mean_file, label_file) {
    //classifier_(model_file, trained_file, mean_file, label_file);
    max_iterations_ = MAX_ITER;
    Q_ = Q;
    overshoot_ = OVERSHOOT;
  }

  void print_vector(const vector<float> vec, size_t init = 0, size_t sz = 30) {

    for (size_t i = init; i != sz; ++i) {
      if (i % 10 == 0)
        std::cout << std::endl;
      std::cout << vec[i] << "  ";
    }

    std::cout << std::endl << std::endl;

  }

  int get_top_classifier(const string& top_label,
                         const vector<string>& labels) {

    vector<string>::const_iterator label_idx =
                                find(labels.begin(), labels.end(), top_label);

    /*
      std::cout << "first size: " << first.size() << "   second size: "
                << second.size() << std::endl;
    */
    CHECK(label_idx != labels.end()) << "(DeepFool) The label was not found";

    // find the top classifier
    return label_idx - labels.begin();

  }

  // TODO: better naming
  int Argmin(const vector<float>& f_p, const vector<float>& w_pn) {

    CHECK(f_p.size() == w_pn.size()) << "(ArgMin) different vector dimensions";
    vector<float> tmp;
    for (size_t i = 0; i < f_p.size(); ++i) {
      tmp.push_back(abs(f_p[i]) / w_pn[i]);
    }

    print_vector(tmp, 200, 400);

    vector<float>::const_iterator min_value = min_element(tmp.begin(), tmp.end());

    CHECK(min_value != tmp.end()) << "(ArgMin) no minimum value; should not occur";

    int ret = min_value - tmp.begin(); //size_t typedef

    return ret;

  }

  // TODO: Find a better way: use transformation() or BLAS (or caffe's BLAS)
  vector<float> subtract_vec(const vector<float>& v1,
                             const vector<float>& v2) {

    CHECK(v1.size() == v2.size())
        << "(DeepFool) Vectors are not of the same size";

    vector<float> result;

    for (size_t i = 0; i < v1.size(); ++i) {
      result.push_back(v1[i] - v2[i]);
    }

    return result;
  }

  // TODO: naming; not actually the norm; move sqrt here?
  float vector_norm(const vector<float>& v) {

    //return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    float sum = 0.0;

    for (size_t i = 0; i != v.size(); ++i) {
      sum += pow(v[i], 2);
    }

    return sqrt(sum);
  }

  // TODO: Remove this function
  void print_img_dim(const cv::Mat& image) {

    std::cout << image.dims << " dimensions and " << std::endl
            << image.size() << std::endl << " and "
            << image.channels() << " channels." << std::endl;

  }



  void DeepFool::adversarial(cv::Mat image, bool PREPROCESSING) {
    // Most classifier_ member functions return vector<vector<float> >
    // objects. This algorithm operates on a single image so we use
    // only the first vector of the vector<vector<>> returned (thus
    // the one in [0] position)


    if (PREPROCESSING) {
      classifier_.Preprocess(image, true);
    }

    // get the most probable prediction (label and
    // output of the classifier)
    Classifier::Prediction top_label_prob =
                           classifier_.Classify(image, 1)[0][0];

    // get classifier's labels
    vector<string> labels = classifier_.get_labels();

    int top_classifier_index = get_top_classifier(top_label_prob.first, labels);


    vector<vector<float> > r;

    cv::Mat x = image;
    int i = 0;

    std::cout << "Before while" << std::endl;
    while (top_label_prob.first == classifier_.Classify(x)[0][0].first &&
           i < max_iterations_) {

      // Just for testing puroses
      // TODO: Remove
      std::cout << "Get top classifier grad" << std::endl;
      std::cout << "True prediction: " << top_label_prob.second << std::endl;
      std::cout << "Current prediction: "
                << classifier_.Classify(x)[0][0].second
                << classifier_.Classify(x)[0][0].first << std::endl;
      std::cout << "2nd best current prediction: "
                << classifier_.Classify(x)[0][1].second
                << classifier_.Classify(x)[0][1].first << std::endl;
      std::cout << "3rd best current predition: "
                << classifier_.Classify(x)[0][2].second
                << classifier_.Classify(x)[0][2].first << std::endl;

      // gradient of the top classifier
      vector<float> top_classifier_grad =
          classifier_.InputGradientofClassifier(x, top_classifier_index)[0];

      //print_vector(top_classifier_grad, 100);

      // The gradients and the output of each classifier
      // TODO: reserve to make the allocation faster?
      vector<vector<float> > w_prime;
      vector<float> f_prime;
      vector<float> w_prime_norm;

      std::cout << "Get predictions" << std::endl;
      // get the predictions
      vector<float> predictions = classifier_.Predict(x)[0];

      std::cout << "Before for" << std::endl;
      int j = 0;
      for (size_t k = 0; k < labels.size(); ++k) {

        if (labels[k] != top_label_prob.first) {

          vector<float> k_classifier_grad =
                        classifier_.InputGradientofClassifier(x, k)[0];

      //    print_vector(k_classifier_grad);
          CHECK(k_classifier_grad.size() == top_classifier_grad.size())
              << "The " << k << "-th classifier and the top classifier are "
              << "not of the same size.";

          // change subvec with a BLAS routine?
          w_prime.push_back(subtract_vec(k_classifier_grad,
                                        top_classifier_grad));

          f_prime.push_back(predictions[k] - top_label_prob.second);

          w_prime_norm.push_back(vector_norm(w_prime[j]));
          ++j;  // TODO: Reduce this by using w_prime.end() - 1
        } else {
          std::cout << "Top label with probability " << top_label_prob.first
                    << " is ignored." << std::endl;
        }
      }
      LOG(INFO) << "f_prime has size: " << f_prime.size();
      LOG(INFO) << "w_prime has size: " << w_prime.size() << " with " << w_prime[0].size();
      LOG(INFO) << "w_prime_norm has size: " << w_prime_norm.size();

      LOG(INFO) << "After for. Before Argmin";

      std::cout << "Printing f_prime" << std::endl;
      print_vector(f_prime, 200, 400);
      std::cout << "Printing w_prime_norm" << std::endl;
      print_vector(w_prime_norm, 200, 400);

      int l_hat = Argmin(f_prime, w_prime_norm);

      LOG(INFO) << "The minimum element is " << l_hat;

      float tmp = abs(f_prime[l_hat]) / std::pow(w_prime_norm[l_hat], 2);

      // TODO: Write this in a more compact form using transform() or
      //       use the linear algebra methods from math_functions.cpp
      for (int j = 0; j < w_prime[l_hat].size(); ++j) {
        w_prime[l_hat][j] *= tmp;
      }

      //print_vector(w_prime[l_hat], 0, w_prime[l_hat].size());

      r.push_back(w_prime[l_hat]);

      LOG(INFO) << "Create r_image";


      cv::Mat r_image(classifier_.get_geometry().height,
                      classifier_.get_geometry().width,
                      CV_32FC3, r[i].data());
      /*
      cv::Mat r_image(r[i], true);
      cv::Size input_size = classifier_.get_geometry();

      std::cout << "Trying to create an image with size: " << input_size.height
                << " x " << input_size.width << std::endl;
      r_image = r_image.reshape(3, input_size.width);
      */

      std::cout << "r_image at iteration " << i << " has " << std::endl;
      print_img_dim(r_image);

      std::cout << "x at iteration " << i << " has " << std::endl;
      print_img_dim(x);

    //  std::cout << "temp_x at iteration " << i << " has " << std::endl;
    //  print_img_dim(temp_x);

      cv::Mat tmp_sum;
      cv::add(x, r_image, tmp_sum);
      x = tmp_sum;

      std::string name = "test_deepfool_x_" +
                          boost::lexical_cast<std::string>(i) + ".jpg";
      cv::Mat tmp_x;
      cv::add(x, classifier_.get_mean(), tmp_x);
      cv::imwrite(name, tmp_x);

      std::string name_2 = "test_deepfool_r_" +
                           boost::lexical_cast<std::string>(i) + ".jpg";
      cv::imwrite(name_2, r_image);

      ++i;
    }

    cv::imwrite("temp_x.jpg", x);

    std::cout << "Process ENDED: " << std::endl;
    std::cout << "True prediction: " << top_label_prob.second << std::endl;
    std::cout << "Current prediction: "
              << classifier_.Classify(x)[0][0].second
              << classifier_.Classify(x)[0][0].first << std::endl;
    std::cout << "2nd best current prediction: "
              << classifier_.Classify(x)[0][1].second
              << classifier_.Classify(x)[0][1].first << std::endl;
    std::cout << "3rd best current predition: "
              << classifier_.Classify(x)[0][2].second
              << classifier_.Classify(x)[0][2].first << std::endl;

    // TODO: Its slow find a better way
    vector<float> vv(r[0].size(), 0.0);

    for (size_t i = 0; i != r.size(); ++i) {
      for (size_t j = 0; j != r[0].size(); ++j) {
        vv[j] += r[i][j];
      }
    }

    print_vector(vv);

    cv::Size input_size = classifier_.get_geometry();
    cv::Mat r_image_final(input_size.width, input_size.height,
                          CV_32FC3, vv.data());
    cv::imwrite("Final_perturbation.jpeg", r_image_final*256);
  }

}
