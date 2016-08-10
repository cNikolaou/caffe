#include <string>
#include <numeric> // inner_product
#include <cmath>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include <boost/lexical_cast.hpp>

#include <caffe/util/math_functions.hpp>
#include <caffe/util/deepfool.hpp>
#include <caffe/util/classifier.hpp>


namespace caffe{

  DeepFool::DeepFool(const std::string& model_file,
                     const std::string& trained_file,
                     const std::string& mean_file,
                     const std::string& label_file,
    //                 const size_t batch_size,
                     const int MAX_ITER, const int Q,
                     const float OVERSHOOT)
              : classifier_(model_file, trained_file, mean_file, label_file) {

    batch_size_ = 256;
    max_iterations_ = MAX_ITER;
    Q_ = Q;
    overshoot_ = OVERSHOOT;
  }

  // TODO: REMOVE this function; used only for verification
  void print_vector(const vector<float> vec, size_t init = 0, size_t sz = 30) {

    for (size_t i = init; i != sz; ++i) {
      if (i % 10 == 0)
        std::cout << std::endl;
      std::cout << vec[i] << "  ";
    }

    std::cout << std::endl << std::endl;

  }

  // TODO: Maybe change the name?
  //       Maybe change the functionality?
  //       Maybe decouple the second functionality?
  // It works as it is, but it might need some changes in the future
  size_t Argmin(const vector<float>& f_p, const vector<float>& w_pn,
                bool RETURN_CLASSIFIER_INDEX=false, int top_classifier_index=-1) {

    CHECK(f_p.size() == w_pn.size()) << "(ArgMin) different vector dimensions";

    vector<float> tmp;
    vector<float> tmp2;
    for (size_t i = 0; i < f_p.size(); ++i) {
      tmp.push_back(std::abs(f_p[i] / w_pn[i]));
      tmp2.push_back((f_p[i]) / w_pn[i]);
    }

    std::cout << "Argmin" << std::endl;
    print_vector(tmp, 200, 700);

    std::cout << "TMP2" << std::endl;
    print_vector(tmp2, 200, 700);

    vector<float>::const_iterator min_value =
                                          min_element(tmp.begin(), tmp.end());

    CHECK(min_value != tmp.end()) << "(ArgMin) no minimum value; should not occur";

    size_t idx = min_value - tmp.begin();

    // Return the classifier index instead of the
    // minimum index in the vectors.
    if (RETURN_CLASSIFIER_INDEX) {
      CHECK(top_classifier_index > 0) << "Please provide the top classifier";
      if (idx >= top_classifier_index) {
        idx++;
      }
    }

    return idx;

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

  // TODO: maybe use BLAS for computation?
  float vector_norm(const vector<float>& v) {

    //return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    float sum = 0.0;

    for (size_t i = 0; i != v.size(); ++i) {
      sum += pow(v[i], 2);
    }

    return sqrt(sum);
  }


  // TODO: REMOVE this function
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


    // in case the image has not been preprocessed
    // before calling the member function
    if (PREPROCESSING) {
      classifier_.Preprocess(image);
    }

    // get classifier's labels
    vector<string> labels = classifier_.get_labels();

    // get the most probable prediction (label and
    // output/probability of the classifier)
    Classifier::Prediction top_label_prob =
                           classifier_.Classify(image, 1)[0][0];

    int top_classifier_index =
                            classifier_.get_label_index(top_label_prob.first);

    // vector that holds the values of the perturbation of each step
    vector<vector<float> > r;

    // image that will change in each iteration by adding the perturbation
    cv::Mat x = image;
    int i = 0;

    // TODO: REMOVE uneccesary std::cout-s
    std::cout << "Before while" << std::endl;

    while (top_label_prob.first == classifier_.Classify(x)[0][0].first &&
           i < max_iterations_) {

      // Just for testing puroses
      // TODO: REMOVE
      std::cout << "Get top classifier grad" << std::endl;
      std::cout << "True prediction: " << top_label_prob.second << std::endl;
      std::cout << "Current prediction: "
                << classifier_.Classify(x)[0][0].second << "   "
                << classifier_.Classify(x)[0][0].first << "   at position "
                << classifier_.get_label_index(classifier_.Classify(x)[0][0].first) << std::endl;
      std::cout << "2nd best current prediction: "
                << classifier_.Classify(x)[0][1].second << "   "
                << classifier_.Classify(x)[0][1].first << "   at position "
                << classifier_.get_label_index(classifier_.Classify(x)[0][1].first) << std::endl;
      std::cout << "3rd best current predition: "
                << classifier_.Classify(x)[0][2].second << "   "
                << classifier_.Classify(x)[0][2].first << "   at position "
                << classifier_.get_label_index(classifier_.Classify(x)[0][2].first) << std::endl;

      // gradient of the top classifier
      vector<float> current_top_classifier_grad =
          classifier_.InputGradientofClassifier(x, top_classifier_index)[0];


      std::cout << "Printing current_top_classifier_grad "<< std::endl;
      print_vector(current_top_classifier_grad, 200, 400);

      //print_vector(current_top_classifier_grad, 100);

      // The gradients and the output of each classifier
      // TODO: reserve to make the allocation faster?
      // TODO: use BLAS to simplify computation?
      vector<vector<float> > w_prime;
      vector<float> f_prime;
      vector<float> w_prime_norm;

      std::cout << "Get predictions" << std::endl;
      // get the predictions
      vector<float> predictions = classifier_.Predict(x)[0];

      std::cout << "Before for" << std::endl;
      int j = 0;
      for (size_t k = 0; k < labels.size(); ++k) {
//      for (size_t k = 200; k < 300; ++k) {
        if (labels[k] != top_label_prob.first) {

          vector<float> k_classifier_grad =
                        classifier_.InputGradientofClassifier(x, k)[0];

      //    print_vector(k_classifier_grad);
          CHECK(k_classifier_grad.size() == current_top_classifier_grad.size())
              << "The " << k << "-th classifier and the top classifier are "
              << "not of the same size.";

          // change subvec with a BLAS routine?
          w_prime.push_back(subtract_vec(k_classifier_grad,
                                        current_top_classifier_grad));

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
      print_vector(f_prime, 200, 700);
      std::cout << "Printing w_prime_norm" << std::endl;
      print_vector(w_prime_norm, 200, 700);

      int l_hat = Argmin(f_prime, w_prime_norm);

      LOG(INFO) << "The minimum element is " << l_hat;

      float tmp = std::abs(f_prime[l_hat]) / std::pow(w_prime_norm[l_hat], 2);

      LOG(INFO) << "tmp value " << tmp;

      // TODO: Write this in a more compact form using transform() or
      //       use the linear algebra methods from math_functions.cpp
      for (int j = 0; j < w_prime[l_hat].size(); ++j) {
        w_prime[l_hat][j] *= tmp;
      }

      std::cout << "Printing w_prime[l_hat] for " << i << std::endl;
      print_vector(w_prime[l_hat], 200, 700);

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

      x = x + (1 + overshoot_) * r_image;

      std::string name = "test_deepfool_x_" +
                          boost::lexical_cast<std::string>(i) + ".jpg";
      cv::Mat tmp_x;
      cv::add(x, classifier_.get_mean(), tmp_x);
      cv::imwrite(name, tmp_x);

      std::string name_2 = "test_deepfool_r_" +
                           boost::lexical_cast<std::string>(i) + ".jpg";
      cv::imwrite(name_2, r_image*256);

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





  void DeepFool::adversarial(std::vector<cv::Mat> images,
                             bool PREPROCESSING) {

    int general_iter = 0;

    if (PREPROCESSING) {
      for (size_t i = 0; i < images.size(); ++i) {
        classifier_.Preprocess(images[i]);
      }
    }

    vector<string> labels = classifier_.get_labels();

    vector<vector<Classifier::Prediction> > tmp_pred =
                                             classifier_.Classify(images,1);

    vector<Classifier::Prediction> top_label_prob;

    for (size_t i = 0; i < images.size(); ++i) {
      top_label_prob.push_back(tmp_pred[i][0]);
    }

    vector<int> top_classifier_index;


    for (size_t i = 0; i < images.size(); ++i) {
      top_classifier_index.push_back(
                        classifier_.get_label_index(top_label_prob[i].first));
    }


    // TODO: REMOVE; For verification only
    for (size_t i = 0; i < images.size(); ++i) {
      std::cout << "top prob " << i << "  "
                << top_label_prob[i].second << " with label "
                << top_label_prob[i].first << " at position "
                << top_classifier_index[i] << std::endl;
    }

    vector<vector<float> > r(images.size(), vector<float>(images[0].total()*images[0].channels(), float()));
    vector<vector<float> > r_final;

    size_t images_processed = 0;

    vector<cv::Mat> x = images;

    // // get at most batch_size_ images from the input
    // for (size_t img_count = 0;
    //      img_count < batch_size_ && img_count < images.size();
    //      ++img_count) {
    //   x.push_back(images[img_count]);
    // }

    vector<cv::Mat> final_image;

    while (images_processed != images.size()) {

      vector<vector<float> > current_top_classifier_grad;

      // TODO: REMOVE; just use all the images in the batch
      // Compute the gradient w.r.t the input of each top classifier
      std::cout << "Computing for " << x.size() << " images" << std::endl;
      for (size_t i = 0; i < x.size(); ++i) {
        current_top_classifier_grad.push_back(
            classifier_.InputGradientofClassifier(x[i],
                                                  top_classifier_index[i])[0]);

        std::cout << "Printing current_top_classifier_grad for " << i << std::endl;
        print_vector(current_top_classifier_grad[i], 200, 400);
      }

      // TODO: For verification
      LOG(INFO) << "current_top_classifier_grad has size: " << current_top_classifier_grad.size();

      for (size_t i = 0; i != x.size(); ++i) {
        LOG(INFO) << "current_top_classifier_grad[" << i << "] has size " << current_top_classifier_grad[i].size();
      }

      vector<vector<float> > f_prime(x.size());//, vector<float>(labels.size()-1, float()));
      vector<vector<float> > w_prime_norm(x.size());//, vector<float>(labels.size()-1, float()));

      vector<vector<float> > predictions = classifier_.Predict(x);
      int testest = 0;
      for (size_t k = 0; k < labels.size(); ++k) {

        vector<vector<float> > k_classifier_grad =
                               classifier_.InputGradientofClassifier(x,k);

        // TODO: Better messages
        CHECK(k_classifier_grad.size() == x.size()) << "ERROR 1";
        CHECK(k_classifier_grad[0].size() ==
              current_top_classifier_grad[0].size()) << "ERROR 2";

        // TODO: REMOVE
      //  LOG(INFO) << "Entering the wierd part";

      // LOG(INFO) << "k_classifier_grad size " << k_classifier_grad.size();
      // LOG(INFO) << "k_classifier_grad[0] size " << k_classifier_grad[0].size();
      // LOG(INFO) << "predictions size " << predictions.size();
      // LOG(INFO) << "predictions[0] size: " << predictions[0].size();

        // for each image in the batch compute the f' and the
        // (normalized) w' needed to compute the minimum perturbation
        for (size_t i = 0; i < x.size(); ++i) {
          if (labels[k] != top_label_prob[i].first) {
            ++testest;
            f_prime[i].push_back(predictions[i][k] - top_label_prob[i].second);

            w_prime_norm[i].push_back(vector_norm(subtract_vec(k_classifier_grad[i],
                                              current_top_classifier_grad[i])));
          } else {
            std::cout << "Not computing for image " << i
                      << " with label " << top_label_prob[i].first
                      << std::endl;
          }
        }

        // TODO: REMOVE
      //  LOG(INFO) << "Leaving the wierd part";
      }

      std::cout << "testest " << testest << std::endl;

      // TODO: For verification
      LOG(INFO) << "f_prime has size: " << f_prime.size();

      for (size_t i = 0; i != x.size(); ++i) {
        LOG(INFO) << "f_prime[" << i << "] has size " << f_prime[i].size();
      }

      LOG(INFO) << "w_prime_norm has size: " << w_prime_norm.size();

      for (size_t i = 0; i != x.size(); ++i) {
        LOG(INFO) << "w_prime_norm[" << i << "] has size " << w_prime_norm[i].size();
      }

      vector<int> l_hat;
      vector<float> tmp;
      vector<vector<float> > w_prime;
      // for each image in the batch
      for (size_t i = 0; i != x.size(); ++i) {

        std::cout << "Printing f_prime for " << i << std::endl;
        print_vector(f_prime[i], 200, 700);

        std::cout << "Printing w_prime_norm for " << i << std::endl;
        print_vector(w_prime_norm[i], 200, 700);

        float ttt = Argmin(f_prime[i], w_prime_norm[i]);//, true, top_classifier_index[i]);
        std::cout << "ttt " << ttt << std::endl;
        l_hat.push_back(ttt);

        tmp.push_back(
              std::abs(f_prime[i][l_hat[i]]) / std::pow(w_prime_norm[i][l_hat[i]],2));

        // TODO: Fix this in a better way; not increasing by one when
        //       recomputing the w_prime returns a wrong result (in case
        //       the top_clasifier, which is removed, is before the l_hat[i])
        if (top_classifier_index[i] <= l_hat[i]) {
          l_hat[i]++;
        }

        w_prime.push_back(
              subtract_vec(
                classifier_.InputGradientofClassifier(x[i], l_hat[i])[0],
                current_top_classifier_grad[i]));

        LOG(INFO) << "l_hat size " << l_hat.size();
        LOG(INFO) << "tmp size " << tmp.size();
        LOG(INFO) << "tmp value " << tmp[i];
        LOG(INFO) << "w_prime size " << w_prime.size();
        LOG(INFO) << "w_prime[i] size: " << w_prime[i].size();
        LOG(INFO) << "r size " << r.size();
        LOG(INFO) << "r[i] size " << r[i].size();
        LOG(INFO) << "Continue";

        for (size_t j = 0; j < w_prime[i].size(); ++j) {
          w_prime[i][j] *= tmp[i];
          r[i][j] += w_prime[i][j];
        }

        std::cout << "Printing w_prime after multi (l_hat) " << i << std::endl;
        print_vector(w_prime[i], 200, 700);

        LOG(INFO) << "Create r_image[" << i << "]";

        cv::Mat r_image(classifier_.get_geometry().height,
                        classifier_.get_geometry().width,
                        CV_32FC3, w_prime[i].data());

        std::cout << "r_image at iteration " << i << " has " << std::endl;
        print_img_dim(r_image);

        std::cout << "x[" << i << "] at iteration " << general_iter << " has " << std::endl;
        print_img_dim(x[i]);

        x[i] = x[i] + (1 + overshoot_) * r_image;

        std::string name = "Image_" +
                            boost::lexical_cast<std::string>(i) +
                            "_iteration_" +
                            boost::lexical_cast<std::string>(general_iter) +
                            ".jpg";
        cv::imwrite(name, x[i] + classifier_.get_mean());

        std::string name2 = "Perturbation_for_image_" +
                            boost::lexical_cast<std::string>(i) +
                            "_iteration_" +
                            boost::lexical_cast<std::string>(general_iter) +
                            ".jpg";
        cv::imwrite(name2, r_image*256);
      }


      int init_size = x.size();
      std::vector<cv::Mat>::iterator itx = x.begin();
      std::vector<std::vector<float> >::iterator itr = r.begin();

      int i = 0;

      // Erase the images whose adversarial perturbations
      // have been found and save the perturbations
      while (itx != x.end()) {
        std::cout << "Image[" << i << "] has prediction "
                  << classifier_.Classify(*itx)[0][0].second << "   "
                  << classifier_.Classify(*itx)[0][0].first << "   at position "
                  << classifier_.get_label_index(classifier_.Classify(*itx)[0][0].first) << std::endl;
        std::cout << "Image[" << i << "] has prediction "
                  << classifier_.Classify(*itx)[0][1].second << "   "
                  << classifier_.Classify(*itx)[0][1].first << "   at position "
                  << classifier_.get_label_index(classifier_.Classify(*itx)[0][1].first) << std::endl;
        std::cout << "Image[" << i << "] has prediction "
                  << classifier_.Classify(*itx)[0][2].second << "   "
                  << classifier_.Classify(*itx)[0][2].first << "   at position "
                  << classifier_.get_label_index(classifier_.Classify(*itx)[0][2].first) << std::endl;

        if (classifier_.Classify(*itx)[0][0].first != top_label_prob[i].first) {
          r_final.push_back(*itr);
          itx = x.erase(itx);
          itr = r.erase(itr);
        } else {
          ++itx;
          ++itr;
        }
        ++i;
      }

      int final_size = init_size - x.size();

      images_processed += final_size;
      std::cout << "Images processed so far: " << images_processed << std::endl;

      ++general_iter;
    }

    for (size_t i = 0; i < r_final.size(); ++i) {
      cv::Mat r_image_final(classifier_.get_geometry().width,
                            classifier_.get_geometry().height,
                            CV_32FC3, r_final[i].data());
      std::string name = "Final_perturbation_" +
                          boost::lexical_cast<std::string>(i) + ".jpg";
      cv::imwrite(name, r_image_final*256);
    }
  }

} // namespace caffe
