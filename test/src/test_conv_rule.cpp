

#include <algorithm>
#include <iostream>
#include <map>
#include "catch.hpp"
#include <prettyprint.h>
#include <string>
#include <vector>
#include <exception>
#include <numeric>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <pybind11_utils.h>
#include <spconv/spconv_ops.h>

namespace py = pybind11;

TEST_CASE("GetConvIndPair", "[SpConvNet]")
{
    
    using namespace py::literals;
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive
    py::exec(R"(
    from __future__ import print_function
    import numpy as np
    import math
    # import spconv
    # import torch

    def get_convolution_output_size(input_size,
                                    kernel_size,
                                    stride,
                                    padding=None,
                                    rate=None):
        ndim = len(input_size)
        if padding is None:
            padding = [0] * ndim

        output_size = []
        for i in range(ndim):
            output_size.append((input_size[i] + 2 * padding[i] - (
                (kernel_size[i] - 1) + 1)) // stride[i] + 1)
        return output_size


    def get_test_sparse_data(shape,
                            num_points,
                            num_channels,
                            integer=False,
                            dtype=np.float32):
        dense_shape = shape
        ndim = len(dense_shape)
        # num_points = np.random.randint(10, 100, size=[batch_size, ndim])
        num_points = np.array(num_points)
        # num_points = np.array([3, 2])
        batch_size = len(num_points)
        batch_indices = []
        coors_total = np.stack(
            np.meshgrid(*[np.arange(0, s) for s in shape]), axis=-1)
        coors_total = coors_total.reshape(-1, ndim)
        for i in range(batch_size):
            np.random.shuffle(coors_total)
            inds_total = coors_total[:num_points[i]]
            inds_total = np.pad(
                inds_total, ((0, 0), (0, 1)), mode="constant", constant_values=i)
            batch_indices.append(inds_total)
        if integer:
            sparse_data = np.random.randint(
                20, 100, size=[num_points.sum(), num_channels]).astype(dtype)
        else:
            sparse_data = np.random.uniform(
                -1, 1, size=[num_points.sum(), num_channels]).astype(dtype)

        # sparse_data = np.arange(1, num_points.sum() + 1).astype(np.float32).reshape(5, 1)
        dense_data = np.zeros(
            [batch_size, num_channels, *dense_shape], dtype=sparse_data.dtype)
        start = 0
        for i, inds in enumerate(batch_indices):
            for j, ind in enumerate(inds):
                dense_slice = (i, slice(None), *ind[:-1])
                dense_data[dense_slice] = sparse_data[start + j]
            start += len(inds)
        batch_indices = np.concatenate(batch_indices, axis=0)
        return {
            "features": sparse_data.astype(dtype),
            "indices": batch_indices.astype(np.int32),
            "features_dense": dense_data.astype(dtype),
        }
    shape = [50, 30, 30]
    num_points = [5000] * 1
    # np.random.seed(np.random.randint(1, 100000))
    in_channels = 64
    sparse_dict = get_test_sparse_data(shape, num_points, in_channels)
    features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)
    indices = np.ascontiguousarray(sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
    features_dense = sparse_dict["features_dense"]
    # indices_t = torch.from_numpy(indices)
    filters = np.random.uniform(0, 1, size=[3, 3, 3, 64, 64]).astype(np.float32)
    # print(outids.shape)
    )");
    SECTION("DebugTest"){

        auto inds = array2TensorView<int>(py::array(py::globals()["indices"]));
        auto inds_tensor = torch::from_blob(inds.data(), {inds.dim(0), inds.dim(1)}, torch::dtype(torch::kInt32));
        auto inds_gpu = inds_tensor.to(torch::Device(torch::kCPU));
        
        auto features = array2TensorView<float>(py::array(py::globals()["features"]));
        auto features_tensor = torch::from_blob(features.data(), {features.dim(0), features.dim(1)}, torch::dtype(torch::kFloat));
        auto features_gpu = features_tensor.to(torch::Device(torch::kCUDA, 0));
        auto filters = array2TensorView<float>(py::array(py::globals()["filters"]));
        auto filters_tensor = torch::from_blob(filters.data(), {filters.dim(0), filters.dim(1), filters.dim(2), filters.dim(3), filters.dim(4)}, torch::dtype(torch::kFloat));
        auto filters_gpu = filters_tensor.to(torch::Device(torch::kCUDA, 0));
        
        auto outputs = spconv::getIndicePair<3>(inds_gpu, 1, {46, 26, 26}, {50, 30, 30}, {3, 3, 3},
            {1, 1, 1}, {0, 0, 0}, {2, 2, 2}, {0, 0, 0}, 0, 0, 0);
        // std::cout << outputs[2] << std::endl;
        /*
        auto output = spconv::indiceConv<float>(features_gpu, filters_gpu, outputs[1], outputs[2], outputs[0].size(0), false);
        std::cout << output << std::endl;*/
    }
}