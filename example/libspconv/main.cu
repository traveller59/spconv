/*
libspconv example for INFERENCE, not TRAINING.
*/

#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>

#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>

#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <tensorview/io/jsonarray.h>
#include <tensorview/parallel/map.h>
using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;
using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;
using ConvTunerSimple =
    spconvlib::spconv::csrc::sparse::convops::spops::ConvTuner;
using ConvGemmOps =
    spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
using SimpleExternalSpconvMatmul =
    spconvlib::spconv::csrc::sparse::convops::SimpleExternalSpconvMatmul;
using InferenceOps = spconvlib::spconv::csrc::sparse::inference::InferenceOps;
using Point2VoxelGPU3D =
    spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;
using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;
using GemmTunerSimple =
    spconvlib::spconv::csrc::sparse::convops::spops::GemmTuner;

int main(int argc, char **argv) {
  bool is_int8 = false;
  float inp_scale = 0.04;
  float out_scale = 0.05;

  tv::ssprint("Hello libspconv!!!");
  TV_ASSERT_RT_ERR(argc == 2, "usage: main /path/to/benchmark-pc.jarr, you can "
                              "find it in example/libspconv.")
  std::string path = argv[1];
  int ndim = 3;
  int max_num_voxels = 200000;
  int num_point_per_voxel = 1;
  std::array<float, 3> vsize_xyz{0.1, 0.1, 0.1};

  Point2VoxelGPU3D p2v{vsize_xyz,
                       {-80, -80, -2, 80, 80, 6},
                       3,
                       max_num_voxels,
                       num_point_per_voxel};
  // load pc from tv::io::JsonArray, can be created from cumm.tensorview.tvio
  auto pc_jarr = tv::io::load_from_file(path);
  auto pc = pc_jarr.tensors.at(0).cuda();
  // in real inference engine, you need to set a upper bound for a point cloud,
  // then pad or truncate your point cloud. ignored in this example.
  auto point_limit = pc.dim(0);
  // use simple method, workspace tensor is created internally in
  // Point2VoxelGPU3D.
  auto p2v_res_simple = p2v.point_to_voxel_hash(pc);
  // or manage workspace by your self (tensorrt)
  // outputs
  tv::Tensor voxels_padded = tv::empty(
      {max_num_voxels, num_point_per_voxel, pc.dim(1)}, tv::float32, 0);
  tv::Tensor indices_padded_no_batch = tv::empty(
      {max_num_voxels, ndim /*batch size not included here*/}, tv::int32, 0);
  tv::Tensor num_per_voxel_padded = tv::empty({max_num_voxels}, tv::int32, 0);
  // workspaces
  tv::Tensor hash_key_value =
      tv::empty({point_limit * 2 /*you should use limit * 2 as hash size*/},
                tv::custom128 /*always use int64*2 as key and value type*/,
                0 /*0 is current GPU, -1 is CPU*/);
  // internal usage
  tv::Tensor point_indice_data = tv::empty({point_limit}, tv::int64, 0);
  // indicate voxel id for every point, -1 if no voxel assigned. used for
  // segmentation models.
  tv::Tensor points_voxel_id = tv::empty({point_limit}, tv::int64, 0);

  auto p2v_res = Point2VoxelGPU3D::point_to_voxel_hash_static(
      pc, voxels_padded, indices_padded_no_batch, num_per_voxel_padded,
      hash_key_value, point_indice_data, points_voxel_id, vsize_xyz,
      tv::arrayops::to_std_array(p2v.grid_size),
      tv::arrayops::to_std_array(p2v.grid_stride),
      tv::arrayops::to_std_array(p2v.coors_range));
  tv::Tensor voxels = std::get<0>(p2v_res).cuda().view(-1, 3);
  int real_num_voxels = voxels.dim(0);
  // add a batch index to previous indices.
  auto indices_padded =
      tv::zeros({indices_padded_no_batch.dim(0), 4}, tv::int32, 0);
  indices_padded.slice(1, 1, 4, 1, false, false)
      .copy_2d_pitched_(indices_padded_no_batch);
  auto indices_cpu = indices_padded.cpu();
  auto indices_cpu_data_ptr = indices_cpu.data_ptr<int32_t>();
  for (int i = 0; i < 5; ++i) {
    auto cur_indices_cpu_data_ptr = indices_cpu_data_ptr + i * 4;
    tv::ssprint(cur_indices_cpu_data_ptr[0], cur_indices_cpu_data_ptr[1],
                cur_indices_cpu_data_ptr[2], cur_indices_cpu_data_ptr[3]);
  }

  auto num_per_voxel = std::get<2>(p2v_res);
  // convert voxels from f32 to f16.
  voxels_padded = voxels_padded.view(-1, pc.dim(1));
  tv::ssprint("num voxels", voxels.shape(), voxels_padded.shape());
  auto voxels_f16_padded = tv::zeros(voxels_padded.shape(), tv::float16, 0);
  auto voxels_f16_ptr = voxels_f16_padded.data_ptr<__half>();
  auto voxels_ptr = voxels_padded.data_ptr<float>();
  tv::kernel_1d_map(0, voxels_f16_padded.size(),
                    [=] TV_GPU_LAMBDA(size_t i) mutable {
                      voxels_f16_ptr[i] = __half(voxels_ptr[i]);
                    });
  // create simple weights and bias.
  std::vector<int32_t> ksize{3, 3, 3};
  std::vector<int32_t> padding{1, 1, 1};
  std::vector<int32_t> dilation{1, 1, 1};
  std::vector<int32_t> stride{1, 1, 1};
  // weight layout is KRSC, [out channels, *ksize, in channels]
  int KV = ksize[0] * ksize[1] * ksize[2]; // kernel volume

  tv::Tensor weights = tv::zeros({64, 3, 3, 3, 3}, tv::float16, 0);
  tv::Tensor bias = tv::zeros({64}, tv::float16, 0);
  // in inference engine, the number of output points must have a upper bound.
  // the shape from point2voxel is input shape.
  auto p2v_grid_size = p2v.get_grid_size();
  std::vector<int32_t> input_dims(p2v_grid_size.begin(), p2v_grid_size.end());
  // calc output dims
  auto out_dims = SpconvOps::get_conv_output_size(input_dims, ksize, stride,
                                                  padding, dilation);
  tv::ssprint(ksize, input_dims, out_dims);

  std::vector<int64_t> output_dims_i64(out_dims.begin(), out_dims.end());
  // if shape is too large, we will use slower int64->int32 hash table instead
  // of int32->int32 table.
  int64_t out_spatial_volume =
      std::accumulate(output_dims_i64.begin(), output_dims_i64.end(),
                      int64_t(1), std::multiplies<int64_t>());
  bool use_int64_hash_k =
      out_spatial_volume >= int64_t(std::numeric_limits<int>::max());
  int static_num_act_in = voxels_padded.dim(0);
  // static_num_act_in is just out_inds_num_limit of previous conv layer.
  // for regular conv, the input tensor has static shape, we should save a CPU
  // variable of real num_act_out. here we just use num_act_in.
  int real_num_act_in = real_num_voxels;
  // you need to slice all static inputs with real_num_act_in in static
  // inference engine, e.g. tensorrt. here we don't need to do that.

  // our algorithm support upper bound
  int out_inds_num_limit = 100000; // upper bound of number of output indices.

  int batch_size = 1;
  // we shouldn't use standard transpose in any code.
  int transpose = false;
  cudaStream_t stream = 0;
  tv::Context ctx;
  ctx.set_cuda_stream_int(reinterpret_cast<std::uintptr_t>(stream));
  // get_compute_capability is very slow, don't forget to cache arch result.
  auto arch = ConvGemmOps::get_compute_capability();
  // remove pad
  auto input_features_real =
      voxels_f16_padded.slice_first_axis(0, real_num_act_in);
  auto input_indices_real = indices_padded.slice_first_axis(0, real_num_act_in);

  for (int i = 0; i < 2; ++i) {
    // if your kernel volume > 32, you need to use
    // tv::gemm::SparseConvAlgo::kNative. otherwise use kMaskImplicitGemm.
    if (i == 0) {
      if (is_int8){
        // native don't support int8
        continue;
      }
      auto conv_algo = tv::gemm::SparseConvAlgo::kNative;
      bool inverse = false;
      // native algo code example
      tv::ssprint("native example");
      for (int j = 0; j < 2; ++j) {
        bool is_subm = j == 0;
        int workspace_size = SpconvOps::get_indice_gen_workspace_size(
            KV, static_num_act_in, out_inds_num_limit,
            0 /*used in implicit gemm*/, is_subm, use_int64_hash_k,
            false /*used in implicit gemm*/);
        // you should return workspace size in tensorrt plugin method.
        tv::Tensor workspace = tv::empty({workspace_size}, tv::uint8, 0);
        // get tensor map required by pair gen from workspace
        // keep in mind that our indice gen function use a "allocator" to alloc
        // temp/out tensors, in python we use TorchAllocator which is a simple
        // dynamic allocator, in c++ (inference engine) we need to use
        // fixed-size workspace and create a static allocator.
        auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
            workspace.raw_data(), KV, static_num_act_in,
            is_subm ? static_num_act_in : out_inds_num_limit, 0, is_subm,
            use_int64_hash_k, false);
        // unlike implicit gemm, native pair isn't tight padded.
        tv::Tensor pair = tv::empty({2, KV, static_num_act_in}, tv::int32, 0);
        tv::Tensor indices_kernel_num = tv::zeros({KV}, tv::int32, 0);
        tv::Tensor out_inds = tv::empty(
            {is_subm ? static_num_act_in : out_inds_num_limit, ndim + 1},
            tv::int32, 0);

        ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair});
        ws_tensors.insert(
            {SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});
        ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});

        StaticAllocator alloc(ws_tensors);

        int num_act_out_real = SpconvOps::get_indice_pairs(
            alloc, input_indices_real, batch_size, out_dims,
            static_cast<int>(tv::gemm::SparseConvAlgo::kNative), ksize, stride,
            padding, dilation, {0, 0, 0}, is_subm, false,
            reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit,
            static_num_act_in);
        tv::Tensor out_features_padded =
            tv::empty({is_subm ? static_num_act_in : out_inds_num_limit, 64},
                      tv::float16, 0);
        auto out_features_real =
            out_features_padded.slice_first_axis(0, num_act_out_real);
        // subm contains a regular gemm
        // so we need to slice input and output
        GemmTunerSimple gemm_tuner(GemmMain::get_all_algo_desp());
        if (is_subm) {
          // unlike regular conv, indices_kernel_num only [0:KV / 2] contains
          // valid numbers, [KV / 2:] is zero so we need to slice input and
          // output with real size to tell indice_conv the center size.
          std::unordered_map<std::string, tv::Tensor> tensor_dict{
              {SPCONV_ALLOC_FEATURES, input_features_real},
              {SPCONV_ALLOC_FILTERS, weights},
              {SPCONV_ALLOC_OUT_FEATURES, out_features_real}};
          StaticAllocator alloc2(tensor_dict);
          // the SimpleExternalSpconvMatmul is used to perform bias operations
          // provided by external bias library such as cublasLt. in pytorch this
          // class use pytorch matmul.
          SimpleExternalSpconvMatmul ext_mm(alloc2);

          ConvGemmOps::indice_conv(
              alloc2, ext_mm, gemm_tuner, true, false, input_features_real,
              weights, pair, indices_kernel_num, arch, out_features_real.dim(0),
              inverse, is_subm,
              static_cast<int>(tv::gemm::SparseConvAlgo::kNative),
              reinterpret_cast<std::uintptr_t>(stream), bias,
              1.0
              /*bias alpha, only used for leaky relu*/,
              0.0, tv::gemm::Activation::kReLU);
        } else {
          // regular conv use numbers in indices_kernel_num to perform gemm
          // so we don't need to slice.
          std::unordered_map<std::string, tv::Tensor> tensor_dict{
              {SPCONV_ALLOC_FEATURES, voxels_f16_padded},
              {SPCONV_ALLOC_FILTERS, weights},
              {SPCONV_ALLOC_OUT_FEATURES, out_features_padded}};
          StaticAllocator alloc2(tensor_dict);
          // the SimpleExternalSpconvMatmul is used to perform bias operations
          // provided by external bias library such as cublasLt. in pytorch this
          // class use pytorch matmul.
          SimpleExternalSpconvMatmul ext_mm(alloc2);
          ConvGemmOps::indice_conv(
              alloc2, ext_mm, gemm_tuner, true, false, voxels_f16_padded,
              weights, pair, indices_kernel_num, arch,
              out_features_padded.dim(0), inverse, is_subm,
              static_cast<int>(tv::gemm::SparseConvAlgo::kNative),
              reinterpret_cast<std::uintptr_t>(stream), bias, 1.0, 0.0,
              tv::gemm::Activation::kReLU);
        }
      }
    } else {
      auto conv_algo = tv::gemm::SparseConvAlgo::kMaskImplicitGemm;
      tv::ssprint("implicit gemm example");
      // implicit gemm code example
      for (int j = 0; j < 2; ++j) {
        bool is_subm = j == 0;
        // direct table: a hash based algorithm that don't need unique. enabled
        // by default.
        bool direct_table = true;
        // only regular conv need direct table.
        bool use_direct_table = direct_table && !is_subm;
        auto max_act_out_theory = SpconvOps::get_handcrafted_max_act_out(
            static_num_act_in, ksize, stride, padding, dilation);
        // query workspace size.
        int workspace_size = SpconvOps::get_indice_gen_workspace_size(
            KV, static_num_act_in, out_inds_num_limit, max_act_out_theory,
            is_subm, use_int64_hash_k, use_direct_table);
        // you should return workspace size in tensorrt plugin method.
        tv::Tensor workspace = tv::empty({workspace_size}, tv::uint8, 0);
        // get tensor map required by pair gen from workspace
        // keep in mind that our indice gen function use a "allocator" to alloc
        // temp/out tensors, in python we use TorchAllocator which is a simple
        // dynamic allocator, in c++ (inference engine) we need to use
        // fixed-size workspace and create a static allocator.
        auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
            workspace.raw_data(), KV, static_num_act_in,
            is_subm ? static_num_act_in : out_inds_num_limit,
            max_act_out_theory, is_subm, use_int64_hash_k, use_direct_table);
        // pair can also have a upper bound.
        // !!!!!IMPORTANT!!!!!!! if you provide a static (padded) pair_fwd and
        // other indice data, the output layout is tight pair_fwd_correct =
        // pair_fwd_padded.view(-1)[:KV * real_pair_size].view(KV,
        // real_pair_size) this valid for pair_fwd, pair_bwd, pair_mask_fwd,
        // pair_mask_bwd, mask_argsort_fwd, mask_argsort_bwd.
        int pair_fwd_size_padded =
            is_subm ? static_num_act_in : out_inds_num_limit;
        tv::Tensor pair_fwd_padded =
            tv::empty({KV, pair_fwd_size_padded}, tv::int32, 0);
        // you can find equivalent python code of following code in python
        // package
        bool is_split_mask =
            conv_algo == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm;
        int mask_count = is_split_mask ? 2 : 1;
        tv::Tensor pair_mask_fwd_padded =
            tv::empty({mask_count, pair_fwd_size_padded}, tv::int32, 0);
        tv::Tensor mask_argsort_fwd_padded =
            tv::empty({mask_count, pair_fwd_size_padded}, tv::int32, 0);
        tv::Tensor out_inds = tv::empty(
            {is_subm ? static_num_act_in : out_inds_num_limit, ndim + 1},
            tv::int32, 0);
        tv::Tensor indices_kernel_num = tv::zeros({KV}, tv::int32, 0);
        std::tuple<tv::Tensor, int> pair_res;
        if (is_subm) {
          // subm out inds equal to input inds, just copy them
          out_inds.copy_(indices_padded, ctx);

          // subm exmaple
          // create output tensors and insert them to static allocator
          // output tensors needed in subm get_indice_pairs_implicit_gemm,
          // saved to static allocator.
          ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair_fwd_padded});
          ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK, pair_mask_fwd_padded});
          ws_tensors.insert(
              {SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort_fwd_padded});
          ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});
          ws_tensors.insert(
              {SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});
          StaticAllocator alloc(ws_tensors);

          pair_res = SpconvOps::get_indice_pairs_implicit_gemm(
              alloc, input_indices_real, batch_size, input_dims,
              static_cast<int>(conv_algo), ksize, stride, padding, dilation,
              {0, 0, 0}, is_subm, transpose, false /*is_train*/,
              reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit,
              tv::CUDAKernelTimer(false), use_direct_table);
          // for subm num_act_out always equal to num_act_in_real

        } else {
          // WARNING be careful with inverse conv, understand python
          // code first. no inverse example here.
          // regular conv need more outputs, used for inversed conv.
          // bwd shape is [KV, static num_act_in (previous num_act_out_bound)]
          tv::Tensor pair_bwd_padded =
              tv::empty({KV, static_num_act_in}, tv::int32, 0);
          tv::Tensor pair_mask_bwd_padded =
              tv::empty({mask_count, static_num_act_in}, tv::int32, 0);
          tv::Tensor mask_argsort_bwd_padded =
              tv::empty({mask_count, static_num_act_in}, tv::int32, 0);

          ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair_fwd_padded});
          ws_tensors.insert({SPCONV_ALLOC_PAIR_BWD, pair_bwd_padded});

          ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK, pair_mask_fwd_padded});
          ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK_BWD, pair_mask_bwd_padded});

          ws_tensors.insert(
              {SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort_fwd_padded});
          ws_tensors.insert(
              {SPCONV_ALLOC_MASK_ARG_SORT_BWD, mask_argsort_bwd_padded});

          ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});
          ws_tensors.insert(
              {SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});

          StaticAllocator alloc(ws_tensors);

          pair_res = SpconvOps::get_indice_pairs_implicit_gemm(
              alloc, input_indices_real, batch_size, input_dims,
              static_cast<int>(conv_algo), ksize, stride, padding, dilation,
              {0, 0, 0}, is_subm, transpose, false /*is_train*/,
              reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit,
              tv::CUDAKernelTimer(false), use_direct_table);
        }
        // after get pair datas, we can start to do real convolution!
        // in static inference engine, you need to split pair-gen and conv to
        // different layers to reuse pair data
        // here we just use previous result.
        int num_act_out_real = std::get<1>(pair_res);

        tv::Tensor out_features =
            tv::empty({is_subm ? static_num_act_in : out_inds_num_limit, 64},
                      tv::float16, 0);
        auto out_features_real =
            out_features.slice_first_axis(0, num_act_out_real);

        bool is_mask_split = pair_mask_fwd_padded.dim(0) > 1;
        int mask_split_cnt = pair_mask_fwd_padded.dim(0);
        tv::Tensor mask_tensor =
            tv::zeros({pair_mask_fwd_padded.dim(0)}, tv::uint32, -1);
        // create split mask
        // currently it's a constant for each algo.
        auto mask_tensor_ptr = mask_tensor.data_ptr<uint32_t>();
        if (is_mask_split) {
          auto kv_div_2 = KV / 2;
          auto remain = KV - kv_div_2;
          uint64_t mask_np_1 = 1;
          uint64_t first = ((mask_np_1 << remain) - 1);
          uint64_t second = ((mask_np_1 << kv_div_2) - 1) << remain;
          mask_tensor_ptr[0] = uint32_t(first);
          mask_tensor_ptr[1] = uint32_t(second);
        } else {
          mask_tensor_ptr[0] = 0xffffffff;
        }
        std::vector<tv::Tensor> pair_mask_splits;
        std::vector<tv::Tensor> mask_argsort_splits;
        size_t real_pair_size = KV * num_act_out_real;
        // keep in mind that pair_mask_fwd_padded is tight padded tensor, so we
        // must get real tensor before use them if inversed conv, use xxxx_bwd
        // here instead of fwd.
        auto pair_fwd_real = pair_fwd_padded.view(-1)
                                 .slice_first_axis(0, KV * num_act_out_real)
                                 .view(KV, num_act_out_real);
        auto pair_mask_fwd_real =
            pair_mask_fwd_padded.view(-1)
                .slice_first_axis(0, mask_split_cnt * num_act_out_real)
                .view(mask_split_cnt, num_act_out_real);
        auto mask_argsort_fwd_real =
            mask_argsort_fwd_padded.view(-1)
                .slice_first_axis(0, mask_split_cnt * num_act_out_real)
                .view(mask_split_cnt, num_act_out_real);

        for (int i = 0; i < mask_split_cnt; ++i) {
          pair_mask_splits.push_back(pair_mask_fwd_real[i]);
          mask_argsort_splits.push_back(mask_argsort_fwd_real[i]);
        }
        // create output tensor allocator
        std::unordered_map<std::string, tv::Tensor> tensor_dict{
            {SPCONV_ALLOC_FEATURES, input_features_real},
            {SPCONV_ALLOC_FILTERS, weights},
            {SPCONV_ALLOC_OUT_FEATURES, out_features}};
        StaticAllocator alloc2(tensor_dict);
        ConvTunerSimple tuner(ConvMain::get_all_conv_algo_desp());
        auto conv_run_status = ConvGemmOps::implicit_gemm(
            alloc2, tuner, input_features_real, weights, pair_fwd_real,
            pair_mask_splits, mask_argsort_splits, num_act_out_real,
            mask_tensor, arch, false, is_subm,
            reinterpret_cast<std::uintptr_t>(stream),
            tv::CUDAKernelTimer(false), false, false, bias,
            1.0 /*bias alpha, only used for leaky relu*/,
            0.0 /*unused for now*/, tv::gemm::Activation::kReLU);
        tv::ssprint("selected conv algo",
                    std::get<1>(conv_run_status).algo_desp.__repr__());
        // FINISH!!!
      }
      // calc maximum number of output points.
    }
  }
  checkCudaErrors(cudaStreamSynchronize(stream));
  // p2v.point_to_voxel_hash()
  return 0;
}