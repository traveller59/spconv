// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <boost/geometry.hpp>
#include <spconv/nms_functor.h>
#include <torch/script.h>
#include <vector>

namespace spconv {

namespace functor {
template <typename T, typename Index>
struct NonMaxSupressionFunctor<tv::CPU, T, Index> {
  Index operator()(const tv::CPU &d, tv::TensorView<Index> keep,
                   tv::TensorView<const T> boxes, T threshold, T eps) {
    auto ndets = boxes.dim(0);
    auto suppressed = std::vector<Index>(ndets);
    auto area = std::vector<T>(ndets);
    for (int i = 0; i < ndets; ++i) {
      area[i] =
          (boxes(i, 2) - boxes(i, 0) + eps) * (boxes(i, 3) - boxes(i, 1) + eps);
    }
    int i, j;
    T xx1, xx2, w, h, inter, ovr;
    int keepNum = 0;
    for (int _i = 0; _i < ndets; ++_i) {
      i = _i;
      if (suppressed[i] == 1)
        continue;
      keep[keepNum] = i;
      keepNum += 1;
      for (int _j = _i + 1; _j < ndets; ++_j) {
        j = _j;
        if (suppressed[j] == 1)
          continue;
        xx2 = std::min(boxes(i, 2), boxes(j, 2));
        xx1 = std::max(boxes(i, 0), boxes(j, 0));
        w = xx2 - xx1 + eps;
        if (w > 0) {
          xx2 = std::min(boxes(i, 3), boxes(j, 3));
          xx1 = std::max(boxes(i, 1), boxes(j, 1));
          h = xx2 - xx1 + eps;
          if (h > 0) {
            inter = w * h;
            ovr = inter / (area[i] + area[j] - inter);
            if (ovr >= threshold)
              suppressed[j] = 1;
          }
        }
      }
    }
    return keepNum;
  }
};

template <typename T, typename Index>
struct rotateNonMaxSupressionFunctor<tv::CPU, T, Index> {
  Index operator()(const tv::CPU &d, tv::TensorView<Index> keep,
                   tv::TensorView<const T> boxCorners,
                   tv::TensorView<const T> standupIoU, T threshold) {
    auto ndets = boxCorners.dim(0);
    auto suppressed = std::vector<Index>(ndets);
    int i, j;
    namespace bg = boost::geometry;
    typedef bg::model::point<T, 2, bg::cs::cartesian> point_t;
    typedef bg::model::polygon<point_t> polygon_t;
    polygon_t poly, qpoly;
    std::vector<polygon_t> poly_inter, poly_union;
    T inter_area, union_area, overlap;
    int keepNum = 0;
    for (int _i = 0; _i < ndets; ++_i) {
      i = _i;
      if (suppressed[i] == 1)
        continue;
      keep[keepNum] = i;
      keepNum += 1;
      for (int _j = _i + 1; _j < ndets; ++_j) {
        j = _j;
        if (suppressed[j] == 1)
          continue;
        if (standupIoU(i, j) <= 0.0)
          continue;
        bg::append(poly, point_t(boxCorners(i, 0, 0), boxCorners(i, 0, 1)));
        bg::append(poly, point_t(boxCorners(i, 1, 0), boxCorners(i, 1, 1)));
        bg::append(poly, point_t(boxCorners(i, 2, 0), boxCorners(i, 2, 1)));
        bg::append(poly, point_t(boxCorners(i, 3, 0), boxCorners(i, 3, 1)));
        bg::append(poly, point_t(boxCorners(i, 0, 0), boxCorners(i, 0, 1)));
        bg::append(qpoly, point_t(boxCorners(j, 0, 0), boxCorners(j, 0, 1)));
        bg::append(qpoly, point_t(boxCorners(j, 1, 0), boxCorners(j, 1, 1)));
        bg::append(qpoly, point_t(boxCorners(j, 2, 0), boxCorners(j, 2, 1)));
        bg::append(qpoly, point_t(boxCorners(j, 3, 0), boxCorners(j, 3, 1)));
        bg::append(qpoly, point_t(boxCorners(j, 0, 0), boxCorners(j, 0, 1)));
        bg::intersection(poly, qpoly, poly_inter);

        if (!poly_inter.empty()) {
          inter_area = bg::area(poly_inter.front());
          bg::union_(poly, qpoly, poly_union);
          if (!poly_union.empty()) { // ignore invalid box
            union_area = bg::area(poly_union.front());
            overlap = inter_area / union_area;
            if (overlap >= threshold)
              suppressed[j] = 1;
            poly_union.clear();
          }
        }
        poly.clear();
        qpoly.clear();
        poly_inter.clear();
      }
    }
    return keepNum;
  }
};

} // namespace functor

#define DECLARE_CPU_T_INDEX(T, Index)                                          \
  template struct functor::NonMaxSupressionFunctor<tv::CPU, T, Index>;         \
  template struct functor::rotateNonMaxSupressionFunctor<tv::CPU, T, Index>;

#define DECLARE_CPU_INDEX(Index)                                               \
  DECLARE_CPU_T_INDEX(float, Index);                                           \
  DECLARE_CPU_T_INDEX(double, Index);

DECLARE_CPU_INDEX(int);
DECLARE_CPU_INDEX(long);

#undef DECLARE_CPU_INDEX
#undef DECLARE_CPU_T_INDEX

} // namespace spconv
