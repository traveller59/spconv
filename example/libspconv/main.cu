#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>

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
using InferenceOps =
    spconvlib::spconv::csrc::sparse::inference::InferenceOps;
using Point2VoxelGPU3D =
    spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;

int main(int argc, char** argv){
    tv::ssprint("Hello libspconv!!!");
    TV_ASSERT_RT_ERR(argc == 2, "usage: main /path/to/benchmark-pc.jarr, you can find it in example/libspconv.")
    std::string path = argv[1];
    Point2VoxelGPU3D p2v{{0.1, 0.1, 0.1}, {-80, -80, -2, 80, 80, 6}, 3, 200000, 1};
    auto pc_jarr = tv::io::load_from_file(path);
    auto pc = pc_jarr.tensors.at(0).cuda();
    // you should use point_to_voxel_hash_static in tensorrt and manage hash data in tensorrt workspace.
    auto p2v_res = p2v.point_to_voxel_hash(pc);
    tv::Tensor voxels = std::get<0>(p2v_res).cuda().view(-1, 3);
    auto indices_without_bs = std::get<1>(p2v_res);
    auto indices = tv::zeros({indices_without_bs.dim(0), 4}, tv::int32, 0);
    indices.slice(1, 1, 4, 1, false, false).copy_2d_pitched_(indices_without_bs);
    auto indices_cpu = indices.cpu();
    auto indices_cpu_data_ptr = indices_cpu.data_ptr<int32_t>();
    for (int i = 0; i < 5; ++i){
        auto cur_indices_cpu_data_ptr = indices_cpu_data_ptr + i * 4;
        tv::ssprint(cur_indices_cpu_data_ptr[0], cur_indices_cpu_data_ptr[1], cur_indices_cpu_data_ptr[2], cur_indices_cpu_data_ptr[3]);
    }
    auto num_per_voxel = std::get<2>(p2v_res);
    tv::ssprint("num voxels", voxels.shape());
    auto voxels_f16 = tv::zeros(voxels.shape(), tv::float16, 0);
    auto voxels_f16_ptr = voxels_f16.data_ptr<__half>();
    auto voxels_ptr = voxels.data_ptr<float>();
    tv::kernel_1d_map(0, voxels_f16.size(), [=]TV_GPU_LAMBDA(size_t i)mutable{
        voxels_f16_ptr[i] = __half(voxels_ptr[i]);
    });
    // out channels, ksize, in channels
    tv::Tensor weights = tv::zeros({64, 3, 3, 3, 3}, tv::float16, 0);
    tv::Tensor bias = tv::zeros({64}, tv::float16, 0);

    int KV = 27;
    int out_inds_num_limit = 100000; // upper bound of number of output indices.
    std::vector<int32_t> ksize{3, 3, 3};
    std::vector<int32_t> padding{1, 1, 1};
    std::vector<int32_t> dilation{1, 1, 1};
    std::vector<int32_t> stride{1, 1, 1};
    int ndim = 3;
    auto p2v_grid_size = p2v.get_grid_size();
    std::vector<int32_t> input_dims(p2v_grid_size.begin(),
                                         p2v_grid_size.end());
    auto out_dims = SpconvOps::get_conv_output_size(input_dims, ksize, stride, padding, dilation);
    tv::ssprint(ksize, input_dims, out_dims);

    std::vector<int64_t> output_dims_i64(out_dims.begin(),
                                         out_dims.end());
    int64_t out_spatial_volume =
        std::accumulate(output_dims_i64.begin(), output_dims_i64.end(),
                        int64_t(1), std::multiplies<int64_t>());
    bool use_int64_hash_k =
        out_spatial_volume >= int64_t(std::numeric_limits<int>::max());
    int num_act_in = voxels.dim(0);
    bool is_subm = true;

    bool direct_table = true;
    int batch_size = 1;
    int transpose = false;
    bool use_direct_table = direct_table && !is_subm;
    auto conv_algo = tv::gemm::SparseConvAlgo::kMaskImplicitGemm;
    auto max_act_out_theory = SpconvOps::get_handcrafted_max_act_out(num_act_in, 
        ksize, stride, padding, dilation);
    
    int workspace_size = SpconvOps::get_indice_gen_workspace_size(
        KV, num_act_in, out_inds_num_limit, max_act_out_theory, is_subm,
        use_int64_hash_k, use_direct_table);
    // you should return workspace size in tensorrt plugin method.
    tv::Tensor workspace = tv::empty({workspace_size}, tv::uint8, 0);
    // get tensor map required by pair gen from workspace
    auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
        workspace.raw_data(), KV, num_act_in, is_subm ? num_act_in : out_inds_num_limit,
        max_act_out_theory, is_subm, use_int64_hash_k, use_direct_table);
    // create output tensors and insert them to static allocator
    int pair_size = is_subm ? num_act_in : out_inds_num_limit;
    tv::Tensor pair_fwd = tv::empty({KV, pair_size}, tv::int32, 0);
    bool is_split_mask =
        conv_algo == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm;
    int mask_count = is_split_mask ? 2 : 1;
    tv::Tensor pair_mask_fwd = tv::empty({mask_count, pair_size}, tv::int32, 0);
    tv::Tensor mask_argsort_fwd = tv::empty({mask_count, pair_size}, tv::int32, 0);
    tv::Tensor out_inds = tv::empty({out_inds_num_limit, ndim + 1}, tv::int32, 0);
    tv::Tensor indices_kernel_num = tv::zeros({KV}, tv::int32, 0);
    cudaStream_t stream = 0;
    ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair_fwd});
    ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK, pair_mask_fwd});
    ws_tensors.insert({SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort_fwd});
    ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});
    ws_tensors.insert({SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});
    StaticAllocator alloc(ws_tensors);
    auto pair_res = SpconvOps::get_indice_pairs_implicit_gemm(
        alloc, indices, batch_size, input_dims, static_cast<int>(conv_algo),
        ksize, stride, padding, dilation, {0, 0, 0},
        is_subm, transpose, false,
        reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit,
        tv::CUDAKernelTimer(false), use_direct_table);
    int num_act_out = std::get<1>(pair_res);
    tv::Tensor out_features = tv::empty({num_act_out, 64}, tv::float16, 0);

    // this function is very slow, don't forget to cache result.
    auto arch = ConvGemmOps::get_compute_capability();
    int kv = pair_fwd.dim(0);
    bool is_mask_split = pair_mask_fwd.dim(0) > 1;
    int mask_split_cnt = pair_mask_fwd.dim(0);
    tv::Tensor mask_tensor =
        tv::zeros({pair_mask_fwd.dim(0)}, tv::uint32, -1);
    auto mask_tensor_ptr = mask_tensor.data_ptr<uint32_t>();
    if (is_mask_split) {
      auto kv_div_2 = kv / 2;
      auto remain = kv - kv_div_2;
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
    for (int i = 0; i < mask_split_cnt; ++i) {
      pair_mask_splits.push_back(
          pair_mask_fwd[i]);
      mask_argsort_splits.push_back(
          mask_argsort_fwd[i]);
    }
    std::unordered_map<std::string, tv::Tensor> tensor_dict{
        {SPCONV_ALLOC_FEATURES, voxels_f16},
        {SPCONV_ALLOC_FILTERS, weights},
        {SPCONV_ALLOC_OUT_FEATURES, out_features}};
    StaticAllocator alloc2(tensor_dict);
    ConvTunerSimple tuner(ConvMain::get_all_conv_algo_desp());
    auto conv_res = ConvGemmOps::implicit_gemm(
        alloc2, tuner, voxels_f16, weights, pair_fwd,
        pair_mask_splits, mask_argsort_splits, num_act_out,
        mask_tensor, arch, false, is_subm,
        reinterpret_cast<std::uintptr_t>(stream), tv::CUDAKernelTimer(false),
        false, false, bias, 1.0,
        0.0, tv::gemm::Activation::kReLU);

    // p2v.point_to_voxel_hash()
    return 0;
}