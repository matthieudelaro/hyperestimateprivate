#include <algorithm>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <string>

#include "caffe/neuron_layers.hpp"
#include <google/protobuf/repeated_field.h>

namespace caffe {

/**
 * @brief Modifies values with mean values
 */
template <typename Dtype>
void TweakFeaturesLayer<Dtype>::alterValues(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::cout << "alter mode" << std::endl;

  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();

  int add_value = this->layer_param_.tweak_features_param().add_value();

  // get the features_mean values from prototxt parameter
  google::protobuf::RepeatedField<double> features_mean_proto = this->layer_param_.tweak_features_param().features_mean();
  std::vector<Dtype> mean(features_mean_proto.size());
  int counter = 0;
  // std::cout << "features_mean_proto.size():" << features_mean_proto.size() << std::endl;
  // std::cout << "mean.size():" << mean.size() << std::endl;
  for (google::protobuf::RepeatedField<double>::iterator it = features_mean_proto.begin();
    it != features_mean_proto.end(); ++it) {
    mean[counter] = *it;
    counter++;
  }

  // modify input data depending on mean
  Dtype factor = 3;
  if (count != mean.size()) {
    std::cout << "Size mismatch : there are " << count << " inputs, but features_mean provided containes " << mean.size() << " values. Thus only " << count << " first values will be exagerated." << std::endl;
  }
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] + // original value
                  add_value; // constant
    if (count < mean.size()) {
      top_data[i] += (mean[i] - bottom_data[i]) * factor; // exagerated difference
    }
    // top_data[i] = bottom_data[i] + add_value;
  }
}


// /**
//  * @brief saves the values in order to compute the mean required by alterValues.
//  * Writes the sums so far.
//  */
// template <typename Dtype>
// void TweakFeaturesLayer<Dtype>::saveAverageValues(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   std::cout << "save mode" << std::endl;
//   static int sampleCounter = 0;
//   static std::ofstream myfile;
//   static std::vector<Dtype> sums; // each element [i] stores the sum of bottom_data[i]

//   const Dtype* bottom_data = bottom[0]->cpu_data();
//   const int count = bottom[0]->count();

//   // if this is the first sample
//   if (sampleCounter == 0) {
//     sums.resize(count); // resize the sums vector properly
//   }
//   myfile.open(this->layer_param_.tweak_features_param().output_file_name().c_str(),  std::ofstream::out | std::ofstream::trunc); // open output file
//   sampleCounter++;


//   // create a converter from double to string
//   // std::ostringstream strs;

//   // write average of previous and current bottom_data to file
//   for (int i = 0; i < count; ++i) {
//     sums[i] += bottom_data[i];
//     myfile << sums[i] / sampleCounter << (i == count-1 ? "\n" : ", ");
//   }
//   myfile.close();
// }

/**
 * @brief saves the values in order to compute the mean required by alterValues.
 * Writes the sums so far.
 */
template <typename Dtype>
void TweakFeaturesLayer<Dtype>::saveValues(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::cout << "save mode" << std::endl;
  static int sampleCounter = 0;
  static std::ofstream myfile;
  static std::vector<Dtype> sums; // each element [i] stores the sum of bottom_data[i]

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();

  // if this is the first sample
  if (sampleCounter == 0) {
    myfile.open(this->layer_param_.tweak_features_param().output_file_name().c_str()); // open output file
    sums.resize(count); // resize the sums vector properly
  }
  sampleCounter++;


  // create a converter from double to string
  // std::ostringstream strs;

  // write bottom_data to file
  for (int i = 0; i < count; ++i) {
    sums[i] += bottom_data[i];
    myfile << sums[i] / sampleCounter << (i == count-1 ? "\n" : ", ");
  }
}

/**
 * @brief called by caffe. Depending on parameter, saves or alters
 */
template <typename Dtype>
void TweakFeaturesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Modes : 0: fast forward (does not alter the data). 1: save. 2: modify
  int mode = this->layer_param_.tweak_features_param().mode();
  if (mode == 0) { // fast forward mode
  } else if (mode == 1) {
    this->saveValues(bottom, top);
  } else if (mode == 2) { // alter mode
    this->alterValues(bottom, top);
  }
  // ReLU version of the function :
  // Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // for (int i = 0; i < count; ++i) {
  //   top_data[i] = std::max(bottom_data[i], Dtype(0))
  //       + negative_slope * std::min(bottom_data[i], Dtype(0));
  // }
}

template <typename Dtype>
void TweakFeaturesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Not implemented because useless for the project. This is still the version of the ReLU layer.
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(TweakFeaturesLayer);
#endif

INSTANTIATE_CLASS(TweakFeaturesLayer);
REGISTER_LAYER_CLASS(TweakFeatures);

}  // namespace caffe
