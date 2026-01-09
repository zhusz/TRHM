// Written with the help of codebase of https://github.com/google/spherical-harmonics, released
// under the following license:
//
// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "default_image.h"

namespace sh {

DefaultImage::DefaultImage(int width, int height) : width_(width), height_(height) {
  pixels_.reset(new Eigen::Array3f[width * height]);
}

int DefaultImage::width() const { return width_; }

int DefaultImage::height() const { return height_; }

Eigen::Array3f DefaultImage::GetPixel(int x, int y) const {
  int index = x + y * width_;
  return pixels_[index];
}

void DefaultImage::SetPixel(int x, int y, const Eigen::Array3f& v) {
  int index = x + y * width_;
  pixels_[index] = v;
}

// std::vector<Eigen::Matrix3f> DefaultImage::cast_to_eigen_matrices() {
  /*
  std::vector<Eigen::MatrixXf> result(3);

  for (int h = 0; h < height(); h++) {
    for (int w = 0; w < width(); w++) {
      for (int c = 0; c < 3; c++) {
      }
    }
  }
  return result;
  */
// }
// void DefaultImage::cast_to_eigen_matrices(std::vector<Eigen::MatrixXf>& out) {
//   int H = height();
//   int W = width();
//   out[0];
//   /*
//   for (int c = 0; c < 3; c++) {
//     // out[c].resize(H, W);
//     out_c = out + c;
//     out_c->resize(H, W);
//   }
//   */
//   /*
//   for (int h = 0; h < H; h++) {
//     for (int w = 0; w < W; w++) {
//       for (int c = 0; c < 3; c++) {
//         // out[c](h, w) = GetPixel(w, h)
//         Eigen::Array3d tmp = GetPixel(w, h);
//         out[c](h, w) = tmp(c);
//       }
//     }
//   }
//   */
// }

}  // namespace sh
