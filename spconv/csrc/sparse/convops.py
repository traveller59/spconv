from typing import Optional
import pccm
from cumm.common import GemmBasicHost, NlohmannJson, TensorView
from cumm.constants import CUMM_CPU_ONLY_BUILD
from cumm.conv.main import ConvMainUnitTest
from cumm.gemm.algospec.core import (_GEMM_MIN_ARCH_TO_ALGO, GemmAlgo,
                                     ShuffleStrideType)
from cumm.gemm.main import GemmMainUnitTest
from spconv.constants import NDIM_DONT_CARE, SPCONV_BWD_SPLITK, AllocKeys
from spconv.core import AlgoHint, ConvAlgo
from spconv.csrc.sparse.gather import GatherCPU

from .alloc import ExternalAllocator
from cumm.common import CompileInfo
from .inference import InferenceOps

class ExternalSpconvMatmul(pccm.Class):
    """a helper class to warp matmul operations
    because we don't want to implement matmul
    (link to cublas/mkl/pytorch) in python package.
    """

    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def indice_conv_init_gemm(self):
        code = pccm.code()
        code.arg("features_n, filters_n", "std::string")
        code.arg("all_weight_is_krsc, is_kc_not_ck", "bool")
        code.arg("kv_center, out_channel", "int")
        code.arg("stream_int", "std::uintptr_t", "0")
        code.raw(f"""
        TV_THROW_RT_ERR("not implemented, override this and use preferred blas!!!");
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def indice_conv_cpu_gemm(self):
        code = pccm.code()
        code.arg("inp_buffer_n, out_buffer_n, filters_n", "std::string")
        code.arg("all_weight_is_krsc, is_kc_not_ck", "bool")
        code.arg("nhot, index", "int")
        code.raw(f"""
        TV_THROW_RT_ERR("not implemented, override this and use preferred cpu blas!!!");
        """)
        return code

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def indice_conv_bwd_init_gemm(self):
        code = pccm.code()
        code.arg("features_n, filters_n, out_bp_n, dfilters_n", "std::string")
        code.arg("all_weight_is_krsc, is_kc_not_ck", "bool")
        code.arg("kv_center", "int")
        code.arg("stream_int", "std::uintptr_t", "0")
        code.raw(f"""
        TV_THROW_RT_ERR("not implemented, override this and use preferred blas!!!");
        """)
        return code.ret("tv::Tensor")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def indice_conv_bwd_cpu_gemm(self):
        code = pccm.code()
        code.arg("inp_buffer_n, out_buffer_n, filters_n, dfilters_n",
                 "std::string")
        code.arg("all_weight_is_krsc, is_kc_not_ck", "bool")
        code.arg("nhot, index", "int")
        code.raw(f"""
        TV_THROW_RT_ERR("not implemented, override this and use preferred cpu blas!!!");
        """)
        return code

class SimpleExternalSpconvMatmul(ExternalSpconvMatmul):
    """implement gemm in cuda via cublasLt. (only support forward)
    should be used with tensorrt plugin.
    """
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView, ExternalAllocator)
        self.build_meta.add_libraries("cublasLt")
        self.add_include("cublasLt.h")
        self.add_member("alloc_", "ExternalAllocator&")
        self.add_member("handle_", "cublasLtHandle_t", "0")

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("alloc", "ExternalAllocator&")
        code.ctor_init("alloc_", "alloc")
        code.raw(f"""
        auto stat = cublasLtCreate(&handle_);
        TV_ASSERT_RT_ERR(CUBLAS_STATUS_SUCCESS == stat, "err");
        """)
        return code 

    @pccm.destructor
    def destructor(self):
        code = pccm.code()
        code.raw(f"""
        if (handle_){{
            cublasLtDestroy(handle_);
        }}
        """)
        return code 

    @pccm.static_function
    def check_cublas_status(self):
        code = pccm.code()
        code.arg("status", "cublasStatus_t")
        code.raw(f"""
        if (status != CUBLAS_STATUS_SUCCESS) {{
            printf("cuBLAS API failed with status %d\\n", status);
            throw std::logic_error("cuBLAS API failed");
        }}
        """)
        return code 

    @pccm.static_function
    def tv_dtype_to_blaslt(self):
        code = pccm.code()
        code.arg("dtype", "tv::DType")
        code.raw(f"""
        switch (dtype) {{
        case tv::float32:
            return CUDA_R_32F;
        case tv::float16:
            return CUDA_R_16F;
        case tv::int32:
            return CUDA_R_32I;
        case tv::int8:
            return CUDA_R_8I;
        case tv::uint32:
            return CUDA_R_32U;
        default:
            return CUDA_R_32F;
        }}
        """)
        return code.ret("decltype(CUDA_R_16F)")

    @pccm.static_function(inline=True)
    def tv_dtype_to_compute(self):
        code = pccm.code()
        code.arg("dtype", "tv::DType")
        with code.macro_if_("CUDART_VERSION >= 11000"):
            code.raw(f"""
            switch (dtype) {{
            case tv::float32:
                return CUBLAS_COMPUTE_32F;
            case tv::float16:
                return CUBLAS_COMPUTE_16F;
            case tv::int32:
                return CUBLAS_COMPUTE_32I;
            case tv::int8:
                return CUBLAS_COMPUTE_32F;
            case tv::uint32:
                return CUBLAS_COMPUTE_32F;
            default:
                return CUBLAS_COMPUTE_32F;
            }}
            """)
        with code.macro_else_():
            code.raw(f"""
            switch (dtype) {{
            case tv::float32:
                return CUDA_R_32F;
            case tv::float16:
                return CUDA_R_16F;
            case tv::int32:
                return CUDA_R_32I;
            case tv::int8:
                return CUDA_R_8I;
            case tv::uint32:
                return CUDA_R_32U;
            default:
                return CUDA_R_32F;
            }}
            """)
        code.macro_endif_()
        return code.ret("decltype(auto)")

    @pccm.static_function
    def matmul_colmajor(self):
        code = pccm.code()
        code.arg("handle", "cublasLtHandle_t")
        code.arg("stream", "cudaStream_t")
        code.arg("a, b, c", "tv::Tensor")
        code.arg("transA, transB", "bool")
        code.raw(f"""
        bool transC = false;
        auto m = a.dim(int(!transA));
        auto k = a.dim(int(transA));
        auto k2 = b.dim(int(!transB));
        auto n = b.dim(int(transB));
        TV_ASSERT_INVALID_ARG(k == k2, "error");
        TV_ASSERT_INVALID_ARG(a.dtype() == b.dtype(), "error");

        tv::TensorShape c_shape;
        if (transC) {{
          c_shape = {{m, n}};
        }} else {{
          c_shape = {{n, m}};
        }}

        if (c.empty()) {{
          c = tv::Tensor(c_shape, a.dtype(), a.device());
        }} else {{
          TV_ASSERT_INVALID_ARG(c.dim(0) == c_shape[0] && c.dim(1) == c_shape[1],
                                "error");
        }}
        cublasLtMatmulDesc_t operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        decltype(CUDA_R_16F) scalarType = CUDA_R_16F;
      #if CUDART_VERSION >= 11000
        decltype(CUBLAS_COMPUTE_32F) computeType = CUBLAS_COMPUTE_32F;
      #endif
        if (a.dtype() == tv::float16 && b.dtype() == tv::float16 &&
            c.dtype() == tv::float16) {{
          scalarType = CUDA_R_16F;
      #if CUDART_VERSION >= 11000
          computeType = CUBLAS_COMPUTE_16F;
      #endif

        }} else if (a.dtype() == tv::float32 && b.dtype() == tv::float32 &&
                  c.dtype() == tv::float16) {{
          scalarType = CUDA_R_32F;
      #if CUDART_VERSION >= 11000
          computeType = CUBLAS_COMPUTE_32F;
      #endif
        }} else if (a.dtype() == tv::float32 && b.dtype() == tv::float32 &&
                  c.dtype() == tv::float32) {{
          scalarType = CUDA_R_32F;
      #if CUDART_VERSION >= 11000
          computeType = CUBLAS_COMPUTE_32F;
      #endif
        }} else if (a.dtype() == tv::float16 && b.dtype() == tv::float16 &&
                  c.dtype() == tv::float32) {{
          scalarType = CUDA_R_32F;
      #if CUDART_VERSION >= 11000
          computeType = CUBLAS_COMPUTE_32F;
      #endif
        }} else {{
          TV_THROW_RT_ERR("unsupported");
        }}
      #if CUDART_VERSION >= 11000
        check_cublas_status(
            cublasLtMatmulDescCreate(&operationDesc, computeType, scalarType));
      #else
        check_cublas_status(cublasLtMatmulDescCreate(&operationDesc, scalarType));
      #endif
        cublasOperation_t transa = !transA ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transb = !transB ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transc = !transC ? CUBLAS_OP_N : CUBLAS_OP_T;

        check_cublas_status(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        check_cublas_status(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
        //   check_cublas_status(cublasLtMatmulDescSetAttribute(
        //       operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transc,
        //       sizeof(transc)));

        check_cublas_status(cublasLtMatrixLayoutCreate(
            &Adesc, tv_dtype_to_blaslt(a.dtype()), transa == CUBLAS_OP_N ? m : k,
            transa == CUBLAS_OP_N ? k : m, a.stride(0)));
        check_cublas_status(cublasLtMatrixLayoutCreate(
            &Bdesc, tv_dtype_to_blaslt(b.dtype()), transb == CUBLAS_OP_N ? k : n,
            transb == CUBLAS_OP_N ? n : k, b.stride(0)));
        //   check_cublas_status(cublasLtMatrixLayoutCreate(
        //       &Cdesc, tv_dtype_to_blaslt(c.dtype()), transc == CUBLAS_OP_N ? m : n,
        //       transc == CUBLAS_OP_N ? n : m, c.dim(0)));
        check_cublas_status(cublasLtMatrixLayoutCreate(
            &Cdesc, tv_dtype_to_blaslt(c.dtype()), m, n, c.stride(0)));

        cublasLtMatmulHeuristicResult_t heuristicResult = {{}};
        cublasLtMatmulPreference_t preference = NULL;

        check_cublas_status(cublasLtMatmulPreferenceCreate(&preference));
        size_t workspaceSize = 0;
        check_cublas_status(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
            sizeof(workspaceSize)));
        int returnedResults = 0;

        check_cublas_status(cublasLtMatmulAlgoGetHeuristic(
            handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
            &heuristicResult, &returnedResults));

        if (returnedResults == 0) {{
          check_cublas_status(CUBLAS_STATUS_NOT_SUPPORTED);
        }}

        int alpha_storage[4];
        int beta_storage[4];
        if (scalarType == CUDA_R_32F) {{
          *(reinterpret_cast<float *>(alpha_storage)) = 1.0f;
          *(reinterpret_cast<float *>(beta_storage)) = 0.0f;
        }} else if (scalarType == CUDA_R_16F) {{
          *(reinterpret_cast<__half *>(alpha_storage)) = __half(1.0f);
          *(reinterpret_cast<__half *>(beta_storage)) = __half(0.0f);
        }} else {{
          TV_THROW_RT_ERR("unsupported");
        }}
        check_cublas_status(cublasLtMatmul(
            handle, operationDesc, alpha_storage, a.const_raw_data(), Adesc, b.const_raw_data(),
            Bdesc, beta_storage, c.raw_data(), Cdesc, c.raw_data(), Cdesc,
            &heuristicResult.algo, nullptr, 0, stream));
        if (preference)
          check_cublas_status(cublasLtMatmulPreferenceDestroy(preference));
        if (Cdesc)
          check_cublas_status(cublasLtMatrixLayoutDestroy(Cdesc));
        if (Bdesc)
          check_cublas_status(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Adesc)
          check_cublas_status(cublasLtMatrixLayoutDestroy(Adesc));
        if (operationDesc)
          check_cublas_status(cublasLtMatmulDescDestroy(operationDesc));
        return;
        """)
        return code

    @pccm.static_function
    def matmul(self):
        code = pccm.code()
        code.arg("handle", "cublasLtHandle_t")
        code.arg("stream", "cudaStream_t")
        code.arg("a, b, c", "tv::Tensor")
        code.arg("transA, transB", "bool")

        code.raw(f"""
        return matmul_colmajor(handle, stream, b, a, c, transB, transA);
        """)
        return code 

    @pccm.member_function
    def indice_conv_init_gemm(self):
        code = pccm.code()
        code.arg("features_n, filters_n", "std::string")
        code.arg("all_weight_is_krsc, is_kc_not_ck", "bool")
        code.arg("kv_center, out_channel", "int")
        code.arg("stream_int", "std::uintptr_t")

        code.raw(f"""
        auto features = alloc_.get_tensor_by_name(features_n);
        auto filters = alloc_.get_tensor_by_name(filters_n);
        TV_ASSERT_RT_ERR(!features.is_cpu(), "only supprt cuda");

        auto out_features = alloc_.empty({pccm.literal(AllocKeys.OutFeatures)}, 
            {{features.dim(0), out_channel}}, features.dtype(), features.device());
        if (!all_weight_is_krsc){{
            filters = filters.view(-1, filters.dim(-2), filters.dim(-1));
            if (!is_kc_not_ck){{
                matmul(handle_, reinterpret_cast<cudaStream_t>(stream_int), 
                    features, filters[kv_center], out_features, false, false);
            }}else{{
                matmul(handle_, reinterpret_cast<cudaStream_t>(stream_int), 
                    features, filters[kv_center], out_features, false, true);
            }}
        }}else{{
            filters = filters.view(out_channel, -1, filters.dim(-1));
            matmul(handle_, reinterpret_cast<cudaStream_t>(stream_int), 
                features, filters.select(1, kv_center), out_features, false, true);
        }}
        return out_features;
        """)
        return code.ret("tv::Tensor")


class GemmTuneResult(pccm.Class, pccm.pybind.PybindClassMixin):

    def __init__(self):
        super().__init__()
        self.add_dependency(GemmBasicHost, TensorView)
        self.add_pybind_member("algo_desp", "tv::gemm::GemmAlgoDesp")
        self.add_pybind_member("arch", "std::tuple<int, int>")
        self.add_pybind_member("splitk", "int")

    @pccm.pybind.mark
    @pccm.member_function
    def is_valid(self):
        code = pccm.code()
        code.raw(f"return splitk > 0 && std::get<0>(arch) > 0;")
        return code.ret("bool")

    @pccm.pybind.mark
    @pccm.constructor
    def defaultctor(self):
        code = pccm.code()
        code.ctor_init("algo_desp", "tv::gemm::GemmAlgoDesp()")
        code.ctor_init("arch", "std::make_tuple(-1, -1)")
        code.ctor_init("splitk", "-1")
        return code

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("algo_desp",
                 "tv::gemm::GemmAlgoDesp",
                 pyanno="cumm.tensorview.gemm.GemmAlgoDesp")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("splitk", "int")
        code.ctor_init("algo_desp", "algo_desp")
        code.ctor_init("arch", "arch")
        code.ctor_init("splitk", "splitk")
        return code


class ConvTuneResult(pccm.Class, pccm.pybind.PybindClassMixin):

    def __init__(self):
        super().__init__()
        self.add_dependency(GemmBasicHost, TensorView)
        self.add_pybind_member("algo_desp", "tv::gemm::ConvAlgoDesp")
        self.add_pybind_member("arch", "std::tuple<int, int>")
        self.add_pybind_member("splitk", "int")

    @pccm.pybind.mark
    @pccm.constructor
    def defaultctor(self):
        code = pccm.code()
        code.ctor_init(
            "algo_desp",
            f"tv::gemm::ConvAlgoDesp({NDIM_DONT_CARE}, tv::gemm::ConvOpType::kForward)"
        )
        code.ctor_init("arch", "std::make_tuple(-1, -1)")
        code.ctor_init("splitk", "-1")
        return code

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("algo_desp",
                 "tv::gemm::ConvAlgoDesp",
                 pyanno="cumm.tensorview.gemm.ConvAlgoDesp")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("splitk", "int")
        code.ctor_init("algo_desp", "algo_desp")
        code.ctor_init("arch", "arch")
        code.ctor_init("splitk", "splitk")
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def is_valid(self):
        code = pccm.code()
        code.raw(f"return splitk > 0 && std::get<0>(arch) > 0;")
        return code.ret("bool")


class GemmTunerSimple(pccm.ParameterizedClass):

    def __init__(self, gemm_cu: Optional[GemmMainUnitTest]):
        super().__init__()
        self.add_dependency(ExternalAllocator, GemmTuneResult, TensorView,
                            GemmBasicHost, CompileInfo)
        if gemm_cu is not None:
            self.add_param_class("gemm", gemm_cu, "GemmMain")
        if not CUMM_CPU_ONLY_BUILD:
            assert gemm_cu is not None
            self.add_include("tensorview/profile/cuda_profiler.h")
        self.add_include("tensorview/utility/tuplehash.h")
        self.add_include("mutex")

        self.add_typedef(
            "static_key_t", "std::tuple<bool, bool, bool, int, "
            "int, int, int>")
        self.add_typedef("algo_cache_key_t", "std::tuple<int, "
                         "int, int, int, int>")

        self.add_member("desps_", "std::vector<tv::gemm::GemmAlgoDesp>")
        self.add_member(
            "static_key_to_desps_",
            "std::unordered_map<static_key_t, std::vector<tv::gemm::GemmAlgoDesp>>"
        )
        self.add_member("prebuilt_names_", "std::unordered_set<std::string>")
        self.add_member("mutex_", "std::mutex")

        self.add_member(
            "nk_forward_cache_, nk_dgrad_cache_, mn_cache_",
            "std::unordered_map<algo_cache_key_t, GemmTuneResult>")

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("desps", "std::vector<tv::gemm::GemmAlgoDesp>")
        code.ctor_init("desps_", "desps")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code
        code.raw(f"""

        for (auto& d : desps){{
            static_key_t static_key = std::make_tuple(d.trans_a(), d.trans_b(), d.trans_c(), d.dtype_a, d.dtype_b,
                d.dtype_c, int(d.shuffle_type));
            auto& vec = static_key_to_desps_[static_key];
            vec.push_back(d);
        }}
        for (auto desp : GemmMain::get_all_algo_desp()){{
            prebuilt_names_.insert(desp.__repr__());
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def get_available_algo_str_from_arch(self):
        code = pccm.code()
        code.arg("arch", "std::tuple<int, int>")
        code.raw(f"""
        std::vector<std::string> res;
        """)
        for i in range(len(_GEMM_MIN_ARCH_TO_ALGO) - 1, -1, -1):
            arch_cur, algos = _GEMM_MIN_ARCH_TO_ALGO[i]
            code.raw(f"""
            auto arch_cur_{i} = std::make_tuple(int({arch_cur[0]}), int({arch_cur[1]}));
            """)
            with code.if_(f"arch >= arch_cur_{i}"):
                for algo in algos:
                    code.raw(f"""
                    res.push_back({pccm.literal(algo)});
                    """)
        code.raw(f"return res;")
        return code.ret("std::vector<std::string>")

    @pccm.pybind.mark
    @pccm.member_function
    def get_all_available(self):
        code = pccm.code()
        code.arg("a, b, c", "tv::Tensor")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("shuffle_type", "int")
        code.arg("use_tf32", "bool", "true")

        code.raw(f"""
        if (trans_c){{
            trans_a = !trans_a;
            trans_b = !trans_b;
            std::swap(trans_a, trans_b);
            std::swap(a, b);
            trans_c = false;
        }}
        // auto avail_algos = get_available_algo_str_from_arch(arch);
        std::vector<tv::gemm::GemmAlgoDesp> finally_algos;
        static_key_t static_key = std::make_tuple(trans_a, trans_b, trans_c, int(a.dtype()),
            int(b.dtype()), int(c.dtype()), shuffle_type);
        if (static_key_to_desps_.find(static_key) == static_key_to_desps_.end()){{
            return finally_algos;
        }}
        auto& desps = static_key_to_desps_.at(static_key);
        for (auto& desp : desps){{
            if (arch < desp.min_arch){{
                continue;
            }}
            if (arch >= std::make_tuple(7, 5) && desp.algo == {pccm.literal(GemmAlgo.Volta.value)}){{
                continue;
            }}
            if (!use_tf32){{
                if (desp.tensorop[0] > 0 && a.dtype() == tv::float32 && b.dtype() == tv::float32){{
                    // tf32 op
                    continue;
                }}
            }}
            auto lda = a.stride(0);
            auto ldb = b.stride(0);
            auto ldc = c.stride(0);
            if (desp.supported_ldx(lda, ldb, ldc)){{
                auto desp2 = desp;
                if (desp.is_nvrtc){{
                    if (!CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){{
                        continue;
                    }}
                }}
                if (!CompileInfo::arch_is_compiled_gemm(arch)){{
                    if (!CompileInfo::gemm_algo_can_use_ptx(desp.min_arch, arch)){{
                        if (CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){{
                            desp2.is_nvrtc = true;
                        }}else{{
                            continue;
                        }}
                    }}
                }}
                finally_algos.push_back(desp2);
            }}
        }}
        std::sort(finally_algos.begin(), finally_algos.end(), [](auto a, auto b){{return a.min_arch > b.min_arch;}});
        return finally_algos;
        """)
        return code.ret("std::vector<tv::gemm::GemmAlgoDesp>",
                        pyanno="List[cumm.tensorview.gemm.GemmAlgoDesp]")

    @pccm.member_function
    def extract_mnk(self):
        code = pccm.code()
        code.arg("a_shape, b_shape", "tv::TensorShape")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("shuffle_type", "int")
        code.arg("a_inds_shape, b_inds_shape, c_inds_shape", "tv::TensorShape")
        code.arg("hint", "int", f"{AlgoHint.NoHint.value}")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code.ret("std::tuple<int, int, int>")

        code.raw(f"""
        std::vector<int64_t> a_shape_vec(a_shape.begin(), a_shape.end());
        std::vector<int64_t> b_shape_vec(b_shape.begin(), b_shape.end());
        std::vector<int64_t> a_inds_shape_vec(a_inds_shape.begin(), a_inds_shape.end());
        std::vector<int64_t> b_inds_shape_vec(b_inds_shape.begin(), b_inds_shape.end());
        std::vector<int64_t> c_inds_shape_vec(c_inds_shape.begin(), c_inds_shape.end());

        return GemmMain::extract_mnk(a_shape_vec, b_shape_vec, trans_a,
                                    trans_b, trans_c,
                                    shuffle_type,
                                    a_inds_shape_vec, b_inds_shape_vec,
                                    c_inds_shape_vec);
        """)
        return code.ret("std::tuple<int, int, int>")

    @pccm.static_function
    def extract_mnk_vector(self):
        code = pccm.code()
        code.arg("a_shape, b_shape", "std::vector<int64_t>")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("shuffle_type", "int")
        code.arg("a_inds_shape, b_inds_shape, c_inds_shape",
                 "std::vector<int64_t>")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code.ret("std::tuple<int, int, int>")
        code.raw(f"""
        return GemmMain::extract_mnk(a_shape, b_shape, trans_a,
                                    trans_b, trans_c,
                                    shuffle_type,
                                    a_inds_shape, b_inds_shape,
                                    c_inds_shape);
        """)
        return code.ret("std::tuple<int, int, int>")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def cached_get_nvrtc_params(self):
        code = pccm.code()
        code.arg("desp",
                 "tv::gemm::GemmAlgoDesp",
                 pyanno="cumm.tensorview.gemm.GemmAlgoDesp")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("stream_int", "std::uintptr_t")

        code.raw(f"""
        TV_THROW_RT_ERR("not implemented in c++, must be overrided in python!!!");
        """)
        return code.ret("tv::gemm::NVRTCParams",
                        pyanno="cumm.tensorview.gemm.NVRTCParams")

    @pccm.pybind.mark
    @pccm.member_function
    def tune_and_cache(self):
        code = pccm.code()

        code.arg("a, b, c", "tv::Tensor")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("shuffle_type", "int")
        code.arg("a_inds, b_inds, c_inds", "tv::Tensor")
        code.arg("hint", "int", f"{AlgoHint.NoHint.value}")
        code.arg("alpha", "float", "1.0")
        code.arg("beta", "float", "0.0")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("num_run", "int", "5")
        code.arg("use_tf32", "bool", "true")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            code.raw("return std::make_tuple(GemmTuneResult(), -1.0f);")
            return code.ret(
                "std::tuple<GemmTuneResult, float>",
                pyanno=
                "Tuple[spconv.core_cc.csrc.sparse.convops.GemmTuneResult, float]"
            )

        code.raw(f"""
        TV_ASSERT_RT_ERR(num_run > 1, "error");
        auto mnk = extract_mnk(a.shape(), b.shape(), trans_a,
                                    trans_b, trans_c,
                                    arch,
                                    shuffle_type,
                                    a_inds.shape(), b_inds.shape(),
                                    c_inds.shape());
        auto m = std::get<0>(mnk);
        auto n = std::get<1>(mnk);
        auto k = std::get<2>(mnk);

        auto avail = get_all_available(a, b, c, trans_a, trans_b, 
            trans_c, arch, shuffle_type, use_tf32);
        auto c_ = c.clone_whole_storage();
        std::vector<GemmTuneResult> all_profile_res;
        std::unordered_set<int> splitk_tests;
        std::vector<float> times;
        float min_time = -1;
        for (auto& desp : avail){{
            tv::gemm::GemmParams params;
            if (desp.is_nvrtc || prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end()){{
                params.nvrtc_params = cached_get_nvrtc_params(desp, arch, stream_int);
            }}
            params.a = a;
            params.b = b;
            params.c = c_;
            params.a_inds = a_inds;
            params.b_inds = b_inds;
            params.c_inds = c_inds;
            params.algo_desp = desp;
            params.alpha = alpha;
            params.beta = beta;
            params.stream = stream_int;
            if (desp.split_k_serial() && (hint & {AlgoHint.BackwardWeight.value})){{
                splitk_tests = {{{', '.join(map(str, SPCONV_BWD_SPLITK))}}};
                splitk_tests.insert(int(a.dim(0)) / std::min(1 << 10, int(a.dim(0))));
                splitk_tests.insert(int(a.dim(0)) / std::min(1 << 11, int(a.dim(0))));
                splitk_tests.insert(int(a.dim(0)) / std::min(1 << 12, int(a.dim(0))));
            }} else {{
                splitk_tests = {{1}};
            }}
            std::vector<int> splitk_tests_vec(splitk_tests.begin(), splitk_tests.end());
            std::sort(splitk_tests_vec.begin(), splitk_tests_vec.end(), [](auto a, auto b){{return a > b;}});
            for (auto spk : splitk_tests_vec){{
                float total_time = 0.0;
                params.split_k_slices = spk;
                int actual_run = 0;
                for (int j = 0; j < num_run; ++j){{
                    auto ev_start = tv::CUDAEvent();
                    auto ev_end = tv::CUDAEvent();
                    ev_start.record(stream_int);
                    GemmMain::matmul2(params);
                    ev_end.record(stream_int);
                    if (j > 0){{
                        // skip first run
                        auto cur_time = tv::CUDAEvent::sync_and_duration(ev_start, ev_end);
                        total_time += cur_time;
                        actual_run++;
                        if (min_time > 0 && cur_time > min_time * 1.5){{
                            // early skip for slow kernels
                            break;
                        }}
                    }}
                }}
                total_time /= actual_run;
                times.push_back(total_time);
                if (min_time < 0){{
                    min_time = total_time;
                }}else{{
                    min_time = std::min(min_time, total_time);
                }}
                all_profile_res.push_back(GemmTuneResult(desp, arch, spk));
            }}
        }}
        TV_ASSERT_RT_ERR(!all_profile_res.empty(), "can't find suitable algorithm");
        auto min_idx = std::min_element(times.begin(), times.end()) - times.begin();
        auto min_tune_res = all_profile_res[min_idx];
        {{
            std::lock_guard<std::mutex> guard(mutex_);
            algo_cache_key_t key;
            if (hint & {AlgoHint.BackwardWeight.value}){{
                key = std::make_tuple(int(a.dtype()), int(b.dtype()), int(c.dtype()), m, n);
                mn_cache_[key] = min_tune_res;
            }}
            else if (hint & {AlgoHint.BackwardInput.value}){{
                key = std::make_tuple(int(a.dtype()), int(b.dtype()), int(c.dtype()), n, k);
                nk_dgrad_cache_[key] = min_tune_res;
            }}
            else if (hint & {AlgoHint.Fowrard.value}){{
                key = std::make_tuple(int(a.dtype()), int(b.dtype()), int(c.dtype()), n, k);
                nk_forward_cache_[key] = min_tune_res;
            }}
            else{{
                TV_THROW_RT_ERR("not implemented");
            }}
        }}
        return std::make_tuple(min_tune_res, times[min_idx]);
        """)
        return code.ret(
            "std::tuple<GemmTuneResult, float>",
            pyanno=
            "Tuple[spconv.core_cc.csrc.sparse.convops.GemmTuneResult, float]")

    @pccm.pybind.mark
    @pccm.member_function
    def get_tuned_algo(self):
        code = pccm.code()
        code.arg("a_dtype, b_dtype, c_dtype", "int")
        code.arg("a_shape, b_shape, c_shape", "std::vector<int64_t>")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("shuffle_type", "int")
        code.arg("a_inds_shape, b_inds_shape, c_inds_shape",
                 "std::vector<int64_t>")
        code.arg("hint", "int", f"{AlgoHint.NoHint.value}")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            code.raw("return std::make_tuple(GemmTuneResult(), false);")
            return code.ret("std::tuple<GemmTuneResult, bool>")

        code.raw(f"""
        auto mnk = GemmMain::extract_mnk(a_shape, b_shape, trans_a,
                                    trans_b, trans_c,
                                    shuffle_type,
                                    a_inds_shape, b_inds_shape,
                                    c_inds_shape);
        auto m = std::get<0>(mnk);
        auto n = std::get<1>(mnk);
        auto k = std::get<2>(mnk);
        GemmTuneResult res;
        bool exists = false;
        {{
            std::lock_guard<std::mutex> guard(mutex_);
            algo_cache_key_t key;
            if (hint & {AlgoHint.BackwardWeight.value}){{
                key = std::make_tuple(int(a_dtype), int(b_dtype), int(c_dtype), m, n);
                if (mn_cache_.find(key) != mn_cache_.end()){{
                    res = mn_cache_.at(key);
                    exists = true;
                }}
            }}
            else if (hint & {AlgoHint.BackwardInput.value}){{
                key = std::make_tuple(int(a_dtype), int(b_dtype), int(c_dtype), n, k);
                if (nk_dgrad_cache_.find(key) != nk_dgrad_cache_.end()){{
                    res = nk_dgrad_cache_.at(key);
                    exists = true;
                }}
            }}
            else if (hint & {AlgoHint.Fowrard.value}){{
                key = std::make_tuple(int(a_dtype), int(b_dtype), int(c_dtype), n, k);
                if (nk_forward_cache_.find(key) != nk_forward_cache_.end()){{
                    res = nk_forward_cache_.at(key);
                    exists = true;
                }}
            }}
            else{{
                TV_THROW_RT_ERR("not implemented");
            }}
        }}
        return std::make_tuple(res, exists);
        """)
        return code.ret("std::tuple<GemmTuneResult, bool>")

    @pccm.pybind.mark
    @pccm.member_function
    def run_with_tuned_result(self):
        code = pccm.code()
        code.arg("profile_res", "GemmTuneResult")
        code.arg("a, b, c", "tv::Tensor")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("stream_int", f"std::uintptr_t")

        code.arg("shuffle_type", "int")
        code.arg("a_inds, b_inds, c_inds", "tv::Tensor")
        code.arg("hint", "int", f"{AlgoHint.NoHint.value}")
        code.arg("alpha", "float", "1.0")
        code.arg("beta", "float", "0.0")

        code.arg("workspace", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)",
                 "cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")
        code.arg("force_nvrtc", f"bool", "false")
        code.arg("bias", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("act_alpha", f"float", "0.0")
        code.arg("act_beta", f"float", "0.0")
        code.arg("act_type", f"tv::gemm::Activation", "tv::gemm::Activation::kNone", "cumm.tensorview.gemm.Activation = Activation.None_")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code

        code.raw(f"""
        auto& desp = profile_res.algo_desp;
        int split_k_slices = 1;
        if (profile_res.splitk > 1){{
            split_k_slices = profile_res.splitk;
        }}

        tv::gemm::GemmParams params;
        bool desp_is_static = prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end();
        if (force_nvrtc || (desp.is_nvrtc || desp_is_static)){{
            params.nvrtc_params = cached_get_nvrtc_params(desp, profile_res.arch, stream_int);
        }}
        params.a = a;
        params.b = b;
        params.c = c;
        params.d = bias;
        params.a_inds = a_inds;
        params.b_inds = b_inds;
        params.c_inds = c_inds;
        params.algo_desp = desp;
        params.split_k_slices = split_k_slices;
        params.stream = stream_int;
        params.alpha = alpha;
        params.beta = beta;
        params.act_alpha = act_alpha;
        params.act_beta = act_beta;
        params.act_type = act_type;

        params.workspace = workspace;
        GemmMain::matmul2(params);
        """)
        return code


class ConvTunerSimple(pccm.ParameterizedClass):

    def __init__(self, conv_cu: Optional[ConvMainUnitTest] = None):
        super().__init__()
        self.add_dependency(ExternalAllocator, ConvTuneResult, TensorView,
                            GemmBasicHost, CompileInfo)
        if conv_cu is not None:
            self.add_param_class("conv", conv_cu, "ConvMain")
        if not CUMM_CPU_ONLY_BUILD:
            assert conv_cu is not None
            self.add_include("tensorview/profile/cuda_profiler.h")

        self.add_include("tensorview/utility/tuplehash.h")
        self.add_include("mutex")

        self.add_typedef("static_key_t",
                         ("std::tuple<int, int, int, int, int, "
                          "int, int, int, int, int>"))
        self.add_typedef(
            "algo_cache_key_t", "std::tuple<int, int, int, int, "
            "int, int, int, int, bool>")

        self.add_member("desps_", "std::vector<tv::gemm::ConvAlgoDesp>")
        self.add_member(
            "static_key_to_desps_",
            "std::unordered_map<static_key_t, std::vector<tv::gemm::ConvAlgoDesp>>"
        )
        self.add_member("prebuilt_names_", "std::unordered_set<std::string>")
        self.add_member("mutex_", "std::mutex")

        self.add_member(
            "kc_forward_cache_, kc_dgrad_cache_, kc_wgrad_cache_",
            "std::unordered_map<algo_cache_key_t, ConvTuneResult>")

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("desps", "std::vector<tv::gemm::ConvAlgoDesp>")
        code.ctor_init("desps_", "desps")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code

        code.raw(f"""
        for (auto& d : desps){{
            static_key_t static_key = std::make_tuple(
                int(d.layout_i), int(d.layout_w), int(d.layout_o),
                d.interleave_i, d.interleave_w, d.interleave_o, d.dtype_input(),
                d.dtype_weight(), d.dtype_output(), int(d.op_type));
            auto& vec = static_key_to_desps_[static_key];
            vec.push_back(d);
        }}
        for (auto desp : ConvMain::get_all_conv_algo_desp()){{
            prebuilt_names_.insert(desp.__repr__());
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def get_available_algo_str_from_arch(self):
        code = pccm.code()
        code.arg("arch", "std::tuple<int, int>")
        code.raw(f"""
        std::vector<std::string> res;
        """)
        for i in range(len(_GEMM_MIN_ARCH_TO_ALGO) - 1, -1, -1):
            arch_cur, algos = _GEMM_MIN_ARCH_TO_ALGO[i]
            code.raw(f"""
            auto arch_cur_{i} = std::make_tuple(int({arch_cur[0]}), int({arch_cur[1]}));
            """)
            with code.if_(f"arch >= arch_cur_{i}"):
                for algo in algos:
                    code.raw(f"""
                    res.push_back({pccm.literal(algo)});
                    """)
        code.raw(f"return res;")
        return code.ret("std::vector<std::string>")

    @pccm.pybind.mark
    @pccm.member_function
    def get_all_available(self):
        code = pccm.code()
        code.arg("inp, weight, out", "tv::Tensor")
        code.arg("layout_i, layout_w, layout_o", "int")
        code.arg("interleave_i, interleave_w, interleave_o", "int")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("op_type", "int")
        code.arg("mask_width", "int")
        code.arg("auto_fp32_accum", "bool")
        code.arg("fp32_accum", "bool")
        code.arg("use_tf32", "bool", "true")
        code.arg("bias", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("scale", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.raw(f"""
        tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);

        bool is_fp16 = (inp.dtype() == tv::float16 && 
            weight.dtype() == tv::float16 && out.dtype() == tv::float16);
        bool use_f32_as_accum = false;
        int kv = 1;
        for (int i = 0; i < weight.ndim() - 2; ++i){{
            kv *= weight.dim(i + 1);
        }}
        if (is_fp16){{
            if (auto_fp32_accum){{
                if (op_type_cpp == tv::gemm::ConvOpType::kForward)
                    use_f32_as_accum = weight.dim(-1) * kv > 128 * 27;
                else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput)
                    use_f32_as_accum = weight.dim(0) * kv > 128 * 27;

            }}else{{
                use_f32_as_accum = fp32_accum;
            }}
        }}
        use_f32_as_accum = false;

        std::vector<tv::gemm::ConvAlgoDesp> finally_algos;
        static_key_t static_key = std::make_tuple(
            layout_i, layout_w, layout_o,
            interleave_i, interleave_w, interleave_o, inp.dtype(),
            weight.dtype(), out.dtype(), op_type);
        if (static_key_to_desps_.find(static_key) == static_key_to_desps_.end()){{
            return finally_algos;
        }}
        auto& desps = static_key_to_desps_.at(static_key);
        for (auto& desp : desps){{
            if (arch < desp.min_arch){{
                continue;
            }}
            if (arch >= std::make_tuple(7, 5) && desp.algo == {pccm.literal(GemmAlgo.Volta.value)}){{
                continue;
            }}
            if (!use_tf32){{
                if (desp.tensorop[0] > 0 && inp.dtype() == tv::float32 && weight.dtype() == tv::float32 && out.dtype() == tv::float32){{
                    // tf32 op
                    continue;
                }}
            }}
            if (arch >= std::make_tuple(7, 0) && is_fp16){{
                // skip simt fp16 kernels if we have tensor core
                if (desp.algo == {pccm.literal(GemmAlgo.Simt.value)}){{
                    continue;
                }}
                if (use_f32_as_accum){{
                    if (desp.dacc == tv::float16){{
                        continue;
                    }}
                }}
            }}

            int ldi = inp.dim(-1);
            int ldw = weight.dim(-1);
            int ldo = out.dim(-1);

            bool mask_width_valid = true;

            if (desp.op_type == tv::gemm::ConvOpType::kBackwardWeight){{
                TV_ASSERT_RT_ERR(mask_width > 0, "eroro");
                mask_width_valid = mask_width % desp.tile_shape[2] == 0;
            }}
            bool require_dynamic_mask = kv > 32;

            if (desp.supported_ldx_conv(ldi, ldw, ldo) && mask_width_valid){{
                if (!bias.empty() && !scale.empty()){{
                    TV_ASSERT_RT_ERR(bias.dtype() == scale.dtype(), "bias/scale dtype must equal to compute dtype in gemm");
                    if (desp.dcomp != bias.dtype()){{
                        continue;
                    }}
                    if (!desp.is_int8_inference){{
                        continue;
                    }}
                }}else{{
                    if (desp.is_int8_inference){{
                        continue;
                    }}
                }}
                auto desp2 = desp;
                if (desp.is_nvrtc){{
                    if (!CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){{
                        continue;
                    }}
                }}
                if (!CompileInfo::arch_is_compiled_gemm(arch)){{
                    if (!CompileInfo::gemm_algo_can_use_ptx(desp.min_arch, arch)){{
                        if (CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){{
                            desp2.is_nvrtc = true;
                        }}else{{
                            continue;
                        }}
                    }}
                }}
                if (require_dynamic_mask){{
                    if (!desp.dynamic_mask){{
                        continue;
                    }}
                }}else{{
                    if (desp.dynamic_mask){{
                        continue;
                    }}
                }}
                finally_algos.push_back(desp2);
            }}
        }}
        std::sort(finally_algos.begin(), finally_algos.end(), [](auto a, auto b){{return a.min_arch > b.min_arch;}});
        return finally_algos;
        """)
        return code.ret("std::vector<tv::gemm::ConvAlgoDesp>",
                        pyanno="List[cumm.tensorview.gemm.ConvAlgoDesp]")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def cached_get_nvrtc_params(self):
        code = pccm.code()
        code.arg("desp",
                 "tv::gemm::ConvAlgoDesp",
                 pyanno="cumm.tensorview.gemm.ConvAlgoDesp")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("stream_int", "std::uintptr_t")

        code.raw(f"""
        TV_THROW_RT_ERR("not implemented in c++, must be overrided in python!!!");
        """)
        return code.ret("tv::gemm::NVRTCParams",
                        pyanno="cumm.tensorview.gemm.NVRTCParams")

    @pccm.pybind.mark
    @pccm.member_function
    def tune_and_cache(self):
        code = pccm.code()
        code.arg("op_type", "int")
        code.arg("inp, weight, output", "tv::Tensor")
        code.arg("layout_i, layout_w, layout_o", "int")
        code.arg("interleave_i, interleave_w, interleave_o", "int")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("mask, mask_argsort, indices", "tv::Tensor")

        code.arg("reverse_mask", "bool")
        code.arg("mask_filter", "uint32_t", "0xffffffff")
        code.arg("mask_width", "int", "-1")
        code.arg("mask_output", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("alpha", "float", "1.0")
        code.arg("beta", "float", "0.0")

        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("auto_fp32_accum", "bool", "true")
        code.arg("fp32_accum", "bool", "false")
        code.arg("num_run", "int", "5")
        code.arg("use_tf32", "bool", "true")
        code.arg("bias", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("scale", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code.ret(
                "std::tuple<ConvTuneResult, float>",
                pyanno=
                "Tuple[spconv.core_cc.csrc.sparse.convops.ConvTuneResult, float]"
            )

        code.raw(f"""
        TV_ASSERT_RT_ERR(num_run > 1, "error");
        auto avail = get_all_available(inp, weight, output, layout_i, layout_w,
                                       layout_o, interleave_i, interleave_w, interleave_o,
                                       arch, op_type, mask_width,
                                       auto_fp32_accum, fp32_accum, use_tf32,
                                       bias, scale);
        inp = inp.clone();
        weight = weight.clone();
        bool need_dynamic_mask = weight.dim(1) > 32;
        output = output.clone();
        int channel_k = output.dim(1);
        int channel_c = inp.dim(1);
        weight = weight.view(channel_k, -1, channel_c);

        std::vector<ConvTuneResult> all_profile_res;
        std::unordered_set<int> splitk_tests;
        std::vector<float> times;
        tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
        float min_time = -1;
        for (auto& desp : avail){{
            tv::gemm::ConvParams params({NDIM_DONT_CARE}, op_type_cpp, tv::CUDAKernelTimer(false));
            if (desp.is_nvrtc || prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end()){{
                params.nvrtc_params = cached_get_nvrtc_params(desp, arch, stream_int);
            }}
            params.conv_algo_desp = desp;
            params.input = inp;
            params.weight = weight.view(channel_k, -1, channel_c);
            params.output = output;

            params.mask_width = mask_width;
            params.alpha = alpha;
            params.beta = beta;
            params.stream = stream_int;
            params.mask_argsort = mask_argsort;
            params.indices = indices;
            params.mask = mask;
            params.mask_output = mask_output;
            if (desp.is_int8_inference){{
                params.bias = bias;
                params.scale = scale;
            }}

            // if (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight){{
            //     TV_ASSERT_RT_ERR(!mask_output.empty(), "error");
            // }}
            if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput){{
                params.reverse_mask = reverse_mask;
            }}
            params.mask_filter = mask_filter;

            if (desp.split_k_serial() && (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight)){{
                splitk_tests = {{{', '.join(map(str, SPCONV_BWD_SPLITK))}}};
                splitk_tests.insert(int(inp.dim(0)) / std::min(1 << 10, int(inp.dim(0))));
                splitk_tests.insert(int(inp.dim(0)) / std::min(1 << 11, int(inp.dim(0))));
                splitk_tests.insert(int(inp.dim(0)) / std::min(1 << 12, int(inp.dim(0))));
            }} else {{
                splitk_tests = {{1}};
            }}
            std::vector<int> splitk_tests_vec(splitk_tests.begin(), splitk_tests.end());
            std::sort(splitk_tests_vec.begin(), splitk_tests_vec.end(), [](auto a, auto b){{return a > b;}});
            for (auto spk : splitk_tests_vec){{
                float total_time = 0.0;
                params.split_k_slices = spk;
                int actual_run = 0;
                for (int j = 0; j < num_run; ++j){{
                    auto ev_start = tv::CUDAEvent();
                    auto ev_end = tv::CUDAEvent();
                    ev_start.record(stream_int);
                    ConvMain::implicit_gemm2(params);
                    ev_end.record(stream_int);
                    if (j > 0){{
                        // skip first run
                        auto cur_time = tv::CUDAEvent::sync_and_duration(ev_start, ev_end);
                        total_time += cur_time;
                        actual_run++;
                        if (min_time > 0 && cur_time > min_time * 1.5){{
                            // early skip for slow kernels
                            break;
                        }}
                    }}
                }}
                total_time /= actual_run;
                times.push_back(total_time);
                if (min_time < 0){{
                    min_time = total_time;
                }}else{{
                    min_time = std::min(min_time, total_time);
                }}
                all_profile_res.push_back(ConvTuneResult(desp, arch, spk));
            }}
        }}
        TV_ASSERT_RT_ERR(!all_profile_res.empty(), "can't find suitable algorithm for", op_type);
        auto min_idx = std::min_element(times.begin(), times.end()) - times.begin();
        auto min_tune_res = all_profile_res[min_idx];
        if (op_type_cpp != tv::gemm::ConvOpType::kBackwardWeight){{
            mask_width = -1;
        }}
        algo_cache_key_t key;
        key = std::make_tuple(int(inp.dtype()), int(weight.dtype()), 
            int(output.dtype()), channel_k, channel_c, std::get<0>(arch), std::get<1>(arch), mask_width, need_dynamic_mask);
        {{
            std::lock_guard<std::mutex> guard(mutex_);

            if (op_type_cpp == tv::gemm::ConvOpType::kForward){{
                kc_forward_cache_[key] = min_tune_res;
            }}
            else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput){{
                kc_dgrad_cache_[key] = min_tune_res;
            }}
            else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight){{
                kc_wgrad_cache_[key] = min_tune_res;
            }}
            else{{
                TV_THROW_RT_ERR("not implemented");
            }}
        }}
        return std::make_tuple(min_tune_res, times[min_idx]);
        """)
        return code.ret(
            "std::tuple<ConvTuneResult, float>",
            pyanno=
            "Tuple[spconv.core_cc.csrc.sparse.convops.ConvTuneResult, float]")

    @pccm.pybind.mark
    @pccm.member_function
    def get_tuned_algo(self):
        code = pccm.code()
        code.arg("op_type", "int")
        code.arg("i_dtype, w_dtype, o_dtype", "int")
        code.arg("k, c", "int")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("mask_width", "int", "-1")
        code.arg("need_dynamic_mask", "bool", "false")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code.ret("std::tuple<ConvTuneResult, bool>")

        code.raw(f"""
        tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
        if (op_type_cpp != tv::gemm::ConvOpType::kBackwardWeight){{
            mask_width = -1;
        }}
        algo_cache_key_t key;
        key = std::make_tuple(i_dtype, w_dtype, o_dtype, k, c, 
            std::get<0>(arch), std::get<1>(arch), mask_width, need_dynamic_mask);
        ConvTuneResult res;
        bool exists = false;
        {{
            std::lock_guard<std::mutex> guard(mutex_);
            if (op_type_cpp == tv::gemm::ConvOpType::kForward){{
                if (kc_forward_cache_.find(key) != kc_forward_cache_.end()){{
                    res = kc_forward_cache_.at(key);
                    exists = true;
                }}
            }}
            else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput){{
                if (kc_dgrad_cache_.find(key) != kc_dgrad_cache_.end()){{
                    res = kc_dgrad_cache_.at(key);
                    exists = true;
                }}

            }}
            else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight){{
                if (kc_wgrad_cache_.find(key) != kc_wgrad_cache_.end()){{
                    res = kc_wgrad_cache_.at(key);
                    exists = true;
                }}
            }}
            else{{
                TV_THROW_RT_ERR("not implemented");
            }}
        }}
        return std::make_tuple(res, exists);
        """)
        return code.ret("std::tuple<ConvTuneResult, bool>")

    @pccm.pybind.mark
    @pccm.member_function
    def run_with_tuned_result(self):
        code = pccm.code()
        code.arg("profile_res", "ConvTuneResult")
        code.arg("op_type", "int")
        code.arg("inp, weight, output", "tv::Tensor")
        code.arg("mask, mask_argsort, mask_output, indices", "tv::Tensor")

        code.arg("reverse_mask", "bool")
        code.arg("mask_filter", "uint32_t", "0xffffffff")
        code.arg("mask_width", "int", "-1")
        code.arg("alpha", "float", "1.0")
        code.arg("beta", "float", "0.0")

        code.arg("stream_int", f"std::uintptr_t", "0")
        code.arg("workspace", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("verbose", f"bool", "false")
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)",
                 "cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(false)")
        code.arg("force_nvrtc", f"bool", "false")
        code.arg("bias", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("act_alpha", f"float", "0.0")
        code.arg("act_beta", f"float", "0.0")
        code.arg("act_type", f"tv::gemm::Activation", "tv::gemm::Activation::kNone", "cumm.tensorview.gemm.Activation = Activation.None_")
        code.arg("scale", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("output_add", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code

        code.raw(f"""
        auto desp = profile_res.algo_desp;
        int split_k_slices = 1;
        if (profile_res.splitk > 1){{
            split_k_slices = profile_res.splitk;
        }}
        int channel_k = output.dim(1);
        int channel_c = inp.dim(1);
        tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
        auto arch = profile_res.arch;
        tv::gemm::ConvParams params({NDIM_DONT_CARE}, op_type_cpp, timer);
        bool desp_is_static = prebuilt_names_.find(desp.__repr__()) == prebuilt_names_.end();
        if (force_nvrtc || (desp.is_nvrtc || desp_is_static)){{
            params.nvrtc_params = cached_get_nvrtc_params(desp, arch, stream_int);
        }}
        params.conv_algo_desp = desp;
        params.input = inp;
        params.weight = weight.view(channel_k, -1, channel_c);
        params.output = output;
        params.verbose = verbose;
        params.bias = bias;
        params.scale = scale;

        params.split_k_slices = split_k_slices;
        params.alpha = alpha;
        params.beta = beta;
        params.act_alpha = act_alpha;
        params.act_beta = act_beta;
        params.act_type = act_type;
        if (!output_add.empty() && desp.is_int8_inference){{
            params.output_add = output_add;
        }}
        params.stream = stream_int;
        params.mask_argsort = mask_argsort;
        params.indices = indices;
        params.mask = mask;

        params.mask_filter = mask_filter;
        params.mask_width = mask_width;
        params.mask_output = mask_output;
        params.reverse_mask = reverse_mask;

        if (timer.enable()){{
            params.timer = timer;
        }}
        params.workspace = workspace;
        ConvMain::implicit_gemm2(params);
        """)
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def query_workspace_size(self):
        code = pccm.code()
        code.arg("desp", "tv::gemm::ConvAlgoDesp")
        code.arg("splitk", "int")
        code.arg("op_type, N, C, K, kv", "int")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code.ret("int")

        code.raw(f'''
        auto mnk = ConvMain::extract_mnk(op_type, N, C, K, kv, -1, -1, true);
        return desp.query_conv_workspace_size(
            std::get<0>(mnk), std::get<1>(mnk), std::get<2>(mnk),
            splitk, kv);
        ''')
        return code.ret("int")


class ConvGemmOps(pccm.ParameterizedClass):

    def __init__(self, gemm_tuner: GemmTunerSimple,
                 conv_tuner: ConvTunerSimple):
        super().__init__()
        self.add_dependency(
            ExternalAllocator,
            GemmTuneResult,
            ConvTuneResult,
            ExternalSpconvMatmul,
            InferenceOps,
        )
        self.add_param_class("gemm", gemm_tuner, "GemmTuner")
        self.add_param_class("conv", conv_tuner, "ConvTuner")

    @pccm.pybind.mark
    @pccm.static_function
    def get_compute_capability(self):
        code = pccm.code()
        code.arg("index", "int", "-1")
        code.raw(f"""
        if (index == -1){{
            checkCudaErrors(cudaGetDevice(&index));
        }}
        #ifdef TV_CUDA
            cudaDeviceProp prop;
            checkCudaErrors(cudaGetDeviceProperties(&prop, index));
            return std::make_tuple(prop.major, prop.minor);
        #else 
            return std::make_tuple(-1, -1);
        #endif
        """)
        return code.ret("std::tuple<int, int>")

    @pccm.pybind.mark
    @pccm.static_function
    def indice_conv(self):
        """1. this function need to take a out features
        that from subm first mm.
        2. this function don't support CPU.
        """
        code = pccm.code()
        code.add_dependency(GatherCPU)

        code.arg("allocator", "ExternalAllocator&")
        code.arg("ext_mm", "ExternalSpconvMatmul&")
        code.arg("gemm_tuner", "GemmTuner&")
        code.arg("all_w_is_krsc, filter_hwio", "bool")

        code.arg("features, filters, indice_pairs", "tv::Tensor")

        code.arg("indice_pair_num", "tv::Tensor")
        code.arg("arch", "std::tuple<int, int>")

        code.arg("num_activate_out", "int")
        code.arg("inverse", "bool", "false")
        code.arg("subm", "bool", "false")
        code.arg("algo", "int", f"{ConvAlgo.Native.value}")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("bias", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("act_alpha", f"float", "0.0")
        code.arg("act_beta", f"float", "0.0")
        code.arg("act_type", f"tv::gemm::Activation", "tv::gemm::Activation::kNone", "cumm.tensorview.gemm.Activation = Activation.None_")
        code.arg("use_tf32", "bool", "true")

        code.raw(f"""
        int kv_dim, out_channel, kv;
        std::vector<int64_t> filter_shape_per_kv;
        bool is_KC_not_CK;
        bool has_bias = !bias.empty();
        bool has_act = act_type != tv::gemm::Activation::kNone;
        if (!all_w_is_krsc){{
            kv_dim = 0;
            is_KC_not_CK = !filter_hwio;
            if (filter_hwio){{
                out_channel = filters.dim(-1);
                filter_shape_per_kv = {{filters.dim(-2), out_channel}};
            }}else{{
                out_channel = filters.dim(-2);
                filter_shape_per_kv = {{out_channel, filters.dim(-1)}};
            }}
            filters = filters.view(-1, filters.dim(-2), filters.dim(-1));
            kv = filters.dim(0);
        }}else{{
            kv_dim = 1;
            out_channel = filters.dim(0);
            filters = filters.view(out_channel, -1, filters.dim(-1));
            is_KC_not_CK = true;
            kv = filters.dim(1);
            filter_shape_per_kv = {{out_channel, filters.dim(-1)}};
        }}
        int kv_center = kv / 2;
        tv::Tensor out_features;
        if (subm){{
            out_features = ext_mm.indice_conv_init_gemm({pccm.literal(AllocKeys.Features)}, 
                {pccm.literal(AllocKeys.Filters)}, all_w_is_krsc,
                is_KC_not_CK, kv_center, out_channel);
        }}else{{
            out_features = allocator.zeros({pccm.literal(AllocKeys.OutFeatures)}, 
                {{num_activate_out, out_channel}}, features.dtype(), features.device(), stream_int);
        }}
        if (has_act || has_bias){{
            TV_ASSERT_RT_ERR(!features.is_cpu(), "bias and act don't support cpu.");
        }}
        if (kv == 1 && subm){{
            if (has_bias && has_act){{
                InferenceOps::bias_add_act_inplace(out_features, bias, act_type, act_alpha, act_beta, stream_int);
            }}else{{
                if (has_bias){{
                    InferenceOps::bias_add_inplace(out_features, bias, stream_int);
                }}
                if (has_act){{
                    InferenceOps::activation_inplace(out_features, act_type, act_alpha, act_beta, stream_int);
                }}
            }}
            return;
        }}
        auto indice_pair_num_cpu = indice_pair_num.cpu();
        auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();
        int maxnhot = 0;
        bool all_zero = true;
        for (int i = 0; i < kv; ++i){{
            if (indice_pair_num_cpu_ptr[i] != 0){{
                indice_pair_num_cpu_ptr[i] = std::min(indice_pair_num_cpu_ptr[i], int(indice_pairs.dim(2)));
                all_zero = false;
                maxnhot = std::max(maxnhot, indice_pair_num_cpu_ptr[i]);
            }}
        }}
        if (subm && all_zero){{
            return;
        }}

        bool inited = subm;
        auto a = features;
        auto c = out_features;
        auto pair_in = indice_pairs[int(inverse)];
        auto pair_out = indice_pairs[int(!inverse)];
        if (features.is_cpu()){{
            TV_ASSERT_RT_ERR(filters.is_cpu() && indice_pairs.is_cpu(), "error");
            auto inp_buffer = allocator.empty({pccm.literal(AllocKeys.InpBuffer)}, 
                {{maxnhot, features.dim(1)}}, features.dtype(), -1);
            auto out_buffer = allocator.empty({pccm.literal(AllocKeys.OutBuffer)}, 
                {{maxnhot, out_features.dim(1)}}, out_features.dtype(), -1);
            for (int i = 0; i < kv; ++i){{
                int nhot = indice_pair_num_cpu_ptr[i];
                if (subm && i == kv_center){{
                    continue;
                }}
                if (subm && i > kv_center){{
                    nhot = indice_pair_num_cpu_ptr[kv - i - 1];
                }}
                if (nhot <= 0){{
                    continue;
                }}
                auto inp_indices = pair_in[i].slice_first_axis(0, nhot);
                auto out_indices = pair_out[i].slice_first_axis(0, nhot);
                GatherCPU::gather(inp_buffer, a, inp_indices);
                ext_mm.indice_conv_cpu_gemm({pccm.literal(AllocKeys.InpBuffer)}, 
                    {pccm.literal(AllocKeys.OutBuffer)},
                    {pccm.literal(AllocKeys.Filters)}, all_w_is_krsc,
                    is_KC_not_CK, nhot, i);
                GatherCPU::scatter_add(c, out_buffer, out_indices);
            }}
            return;
        }}

        """)
        if CUMM_CPU_ONLY_BUILD:
            return code
        code.raw(f"""
        int profile_idx = kv_center;
        if (subm)
            profile_idx = kv_center - 1;
        int nhot_profile = indice_pair_num_cpu_ptr[profile_idx];
        if (nhot_profile == 0){{
            profile_idx = 0;
            for (int i = 0; i < kv; ++i){{
                int nhot = indice_pair_num_cpu_ptr[i];
                if (nhot > nhot_profile){{
                    nhot_profile = nhot;
                    profile_idx = i;
                }}
            }}
        }}
        TV_ASSERT_RT_ERR(nhot_profile > 0, "this shouldn't happen");
        // auto arch = get_compute_capability();
        auto a_shape = a.shape();
        auto c_shape = c.shape();
        int sac_shuffle_type = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAC);
        auto tuned_res_exist = gemm_tuner.get_tuned_algo(
            int(a.dtype()),
            int(filters.dtype()),
            int(c.dtype()),
            std::vector<int64_t>(a_shape.begin(), a_shape.end()),
            filter_shape_per_kv,
            std::vector<int64_t>(c_shape.begin(), c_shape.end()),
            false,
            is_KC_not_CK,
            false,
            arch,
            sac_shuffle_type,
            {{nhot_profile}},
            {{}},
            {{nhot_profile}},
            {AlgoHint.Fowrard.value});
        auto tune_res = std::get<0>(tuned_res_exist);
        auto exists = std::get<1>(tuned_res_exist);

        if (!exists){{
            auto inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile);
            auto out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile);
            auto filter = filters.select(kv_dim, profile_idx);
            auto tune_res_time = gemm_tuner.tune_and_cache(
                a,
                filter,
                c,
                false,
                is_KC_not_CK,
                false,
                arch,
                sac_shuffle_type,
                inp_indices,
                tv::Tensor(),
                out_indices,
                {AlgoHint.Fowrard.value},
                1.0,
                0.0,
                stream_int,
                5, // num_run
                use_tf32);
            tune_res = std::get<0>(tune_res_time);
        }}

        for (int i = 0; i < kv; ++i){{
            int nhot = indice_pair_num_cpu_ptr[i];
            if (subm && i == kv_center){{
                continue;
            }}
            if (subm && i > kv_center){{
                nhot = indice_pair_num_cpu_ptr[kv - i - 1];
            }}
            if (nhot <= 0){{
                continue;
            }}
            auto inp_indices = pair_in[i].slice_first_axis(0, nhot);
            auto out_indices = pair_out[i].slice_first_axis(0, nhot);
            auto b = filters.select(kv_dim, i);
            float beta = inited ? 1.0 : 0.0;
            gemm_tuner.run_with_tuned_result(
                tune_res,
                a,
                b,
                c,
                false,
                is_KC_not_CK,
                false,
                arch,
                stream_int,
                sac_shuffle_type,
                inp_indices,
                tv::Tensor(),
                out_indices,
                {AlgoHint.Fowrard.value},
                1.0,
                beta);
            inited = true;
        }}
        if (has_bias && has_act){{
            InferenceOps::bias_add_act_inplace(out_features, bias, act_type, act_alpha, act_beta, stream_int);
        }}else{{
            if (has_bias){{
                InferenceOps::bias_add_inplace(out_features, bias, stream_int);
            }}
            if (has_act){{
                InferenceOps::activation_inplace(out_features, act_type, act_alpha, act_beta, stream_int);
            }}
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def indice_conv_backward(self):
        code = pccm.code()
        code.add_dependency(GatherCPU)

        code.arg("allocator", "ExternalAllocator&")
        code.arg("ext_mm", "ExternalSpconvMatmul&")
        code.arg("gemm_tuner", "GemmTuner&")
        code.arg("all_w_is_krsc, filter_hwio", "bool")
        code.arg("features, filters, out_bp, indice_pairs", "tv::Tensor")
        code.arg("indice_pair_num", "tv::Tensor")
        code.arg("arch", "std::tuple<int, int>")

        code.arg("inverse", "bool", "false")
        code.arg("subm", "bool", "false")
        code.arg("algo", "int", f"{ConvAlgo.Native.value}")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("use_tf32", "bool", "true")

        code.raw(f"""
        int kv_dim, out_channel, kv;
        std::vector<int64_t> filter_shape_per_kv;
        auto prev_filter_shape_vec = filters.shape_vector();
        bool is_KC_not_CK;
        
        if (!all_w_is_krsc){{
            kv_dim = 0;
            is_KC_not_CK = !filter_hwio;
            if (filter_hwio){{
                out_channel = filters.dim(-1);
                filter_shape_per_kv = {{filters.dim(-2), out_channel}};
            }}else{{
                out_channel = filters.dim(-2);
                filter_shape_per_kv = {{out_channel, filters.dim(-1)}};
            }}
            filters = filters.view(-1, filters.dim(-2), filters.dim(-1));
            kv = filters.dim(0);
        }}else{{
            kv_dim = 1;
            out_channel = filters.dim(0);
            filters = filters.view(out_channel, -1, filters.dim(-1));
            is_KC_not_CK = true;
            kv = filters.dim(1);
            filter_shape_per_kv = {{out_channel, filters.dim(-1)}};
        }}
        int kv_center = kv / 2;
        tv::Tensor din;
        auto dfilters = allocator.zeros({pccm.literal(AllocKeys.DFilters)}, 
                prev_filter_shape_vec, features.dtype(), features.device(), stream_int);
        dfilters = dfilters.view(filters.shape());
        if (subm){{
            din = ext_mm.indice_conv_bwd_init_gemm({pccm.literal(AllocKeys.Features)}, 
                {pccm.literal(AllocKeys.Filters)}, {pccm.literal(AllocKeys.OutBp)},
                {pccm.literal(AllocKeys.DFilters)},
                all_w_is_krsc,
                is_KC_not_CK, kv_center);
        }}else{{
            din = allocator.zeros({pccm.literal(AllocKeys.DIn)}, 
                    features.shape_vector(), features.dtype(), features.device(), stream_int);
        }}
        if (kv == 1 && subm){{
            return;
        }}
        auto indice_pair_num_cpu = indice_pair_num.cpu();
        auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();
        int maxnhot = 0;
        bool all_zero = true;
        for (int i = 0; i < kv; ++i){{
            if (indice_pair_num_cpu_ptr[i] != 0){{
                indice_pair_num_cpu_ptr[i] = std::min(indice_pair_num_cpu_ptr[i], int(indice_pairs.dim(2)));
                all_zero = false;
                maxnhot = std::max(maxnhot, indice_pair_num_cpu_ptr[i]);
            }}
        }}
        if (subm && all_zero){{
            return;
        }}
        bool inited = subm;
        auto pair_in = indice_pairs[int(inverse)];
        auto pair_out = indice_pairs[int(!inverse)];

        if (features.is_cpu()){{
            TV_ASSERT_RT_ERR(filters.is_cpu() && indice_pairs.is_cpu(), "error");
            auto inp_buffer = allocator.empty({pccm.literal(AllocKeys.InpBuffer)}, 
                {{maxnhot, features.dim(1)}}, features.dtype(), -1);
            auto out_buffer = allocator.empty({pccm.literal(AllocKeys.OutBuffer)}, 
                {{maxnhot, out_bp.dim(1)}}, out_bp.dtype(), -1);
            for (int i = 0; i < kv; ++i){{
                int nhot = indice_pair_num_cpu_ptr[i];
                if (subm && i == kv_center){{
                    continue;
                }}
                if (subm && i > kv_center){{
                    nhot = indice_pair_num_cpu_ptr[kv - i - 1];
                }}
                if (nhot <= 0){{
                    continue;
                }}
                auto inp_indices = pair_in[i].slice_first_axis(0, nhot);
                auto out_indices = pair_out[i].slice_first_axis(0, nhot);
                GatherCPU::gather(inp_buffer, features, inp_indices);
                GatherCPU::gather(out_buffer, out_bp, out_indices);
                ext_mm.indice_conv_bwd_cpu_gemm({pccm.literal(AllocKeys.InpBuffer)}, 
                    {pccm.literal(AllocKeys.OutBuffer)}, 
                    {pccm.literal(AllocKeys.Filters)},
                    {pccm.literal(AllocKeys.DFilters)}, all_w_is_krsc,
                    is_KC_not_CK, nhot, i);
                GatherCPU::scatter_add(din, inp_buffer, inp_indices);
            }}
            return;
        }}
        """)
        if CUMM_CPU_ONLY_BUILD:
            return code
        code.raw(f"""

        int profile_idx = kv_center;
        if (subm)
            profile_idx = kv_center - 1;
        int nhot_profile = indice_pair_num_cpu_ptr[profile_idx];
        if (nhot_profile == 0){{
            profile_idx = 0;
            for (int i = 0; i < kv; ++i){{
                int nhot = indice_pair_num_cpu_ptr[i];
                if (nhot > nhot_profile){{
                    nhot_profile = nhot;
                    profile_idx = i;
                }}
            }}
        }}
        TV_ASSERT_RT_ERR(nhot_profile > 0, "this shouldn't happen");
        // auto arch = get_compute_capability();
        int sac_shuffle_type = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAC);
        int sab_shuffle_type = static_cast<int>(tv::gemm::ShuffleStrideType::kShuffleAB);

        auto dgrad_tuned_res_exist = gemm_tuner.get_tuned_algo(
            int(out_bp.dtype()),
            int(filters.dtype()),
            int(din.dtype()),
            out_bp.shape_vector(),
            filter_shape_per_kv,
            din.shape_vector(),
            false,
            !is_KC_not_CK,
            false,
            arch,
            sac_shuffle_type,
            {{nhot_profile}},
            {{}},
            {{nhot_profile}},
            {AlgoHint.BackwardInput.value});
        auto tuned_res_dgrad = std::get<0>(dgrad_tuned_res_exist);
        auto dgrad_exists = std::get<1>(dgrad_tuned_res_exist);
        if (!dgrad_exists){{
            auto inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile);
            auto out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile);
            auto filter = filters.select(kv_dim, profile_idx);
            auto tune_res_time = gemm_tuner.tune_and_cache(
                out_bp,
                filter,
                din,
                false,
                !is_KC_not_CK,
                false,
                arch,
                sac_shuffle_type,
                out_indices,
                tv::Tensor(),
                inp_indices,
                {AlgoHint.BackwardInput.value},
                1.0,
                0.0,
                stream_int,
                5, // num_run
                use_tf32);
            tuned_res_dgrad = std::get<0>(tune_res_time);
        }}
        tv::Tensor a_wgrad, b_wgrad;
        if (is_KC_not_CK){{
            a_wgrad = out_bp;
            b_wgrad = features;
        }}
        else{{
            a_wgrad = features;
            b_wgrad = out_bp;
        }}

        auto wgrad_tuned_res_exist = gemm_tuner.get_tuned_algo(
            int(a_wgrad.dtype()),
            int(b_wgrad.dtype()),
            int(filters.dtype()),
            a_wgrad.shape_vector(),
            b_wgrad.shape_vector(),
            filter_shape_per_kv,
            true,
            false,
            false,
            arch,
            sab_shuffle_type,
            {{nhot_profile}},
            {{nhot_profile}},
            {{}},
            {AlgoHint.BackwardWeight.value});
        auto tuned_res_wgrad = std::get<0>(wgrad_tuned_res_exist);
        auto wgrad_exists = std::get<1>(wgrad_tuned_res_exist);
        if (!wgrad_exists){{
            auto inp_indices = pair_in[profile_idx].slice_first_axis(0, nhot_profile);
            auto out_indices = pair_out[profile_idx].slice_first_axis(0, nhot_profile);
            auto dfilter = dfilters.select(kv_dim, profile_idx);
            tv::Tensor a_inds_wgrad, b_inds_wgrad;
            if (is_KC_not_CK){{
                a_inds_wgrad = out_indices;
                b_inds_wgrad = inp_indices;
            }}else{{
                a_inds_wgrad = inp_indices;
                b_inds_wgrad = out_indices;
            }}
            auto tune_res_time = gemm_tuner.tune_and_cache(
                a_wgrad,
                b_wgrad,
                dfilter,
                true,
                false,
                false,
                arch,
                sab_shuffle_type,
                a_inds_wgrad,
                b_inds_wgrad,
                tv::Tensor(),
                {AlgoHint.BackwardWeight.value},
                1.0,
                0.0,
                stream_int,
                5, // num_run
                use_tf32);
            tuned_res_wgrad = std::get<0>(tune_res_time);
        }}
        
        
        std::vector<int64_t> a_shape{{maxnhot, out_bp.dim(1)}};
        std::vector<int64_t> b_shape{{maxnhot, features.dim(1)}};
        if (!is_KC_not_CK){{
            std::swap(a_shape, b_shape);
        }}
        auto mnk = GemmTuner::extract_mnk_vector(a_shape, b_shape, 
            tuned_res_wgrad.algo_desp.trans_a(),
            tuned_res_wgrad.algo_desp.trans_b(),
            tuned_res_wgrad.algo_desp.trans_c(),
            sab_shuffle_type, 
            {{maxnhot}}, {{maxnhot}}, {{}});
        
        auto ws_size = tuned_res_wgrad.algo_desp.query_workspace_size(
            std::get<0>(mnk), std::get<1>(mnk), std::get<2>(mnk), tuned_res_wgrad.splitk);
        ExternalAllocator::guard_t workspace_guard;
        tv::Tensor workspace;
        if (ws_size > 0){{
            workspace_guard = allocator.empty_guard({{int64_t(ws_size)}}, tv::uint8, 0);
            workspace = workspace_guard->tensor;
        }}

        for (int i = 0; i < kv; ++i){{
            int nhot = indice_pair_num_cpu_ptr[i];
            if (subm && i == kv_center){{
                continue;
            }}
            if (subm && i > kv_center){{
                nhot = indice_pair_num_cpu_ptr[kv - i - 1];
            }}
            if (nhot <= 0){{
                continue;
            }}
            auto inp_indices = pair_in[i].slice_first_axis(0, nhot);
            auto out_indices = pair_out[i].slice_first_axis(0, nhot);
            auto filter_i = filters.select(kv_dim, i);
            float beta = inited ? 1.0 : 0.0;

            gemm_tuner.run_with_tuned_result(
                tuned_res_dgrad,
                out_bp,
                filter_i,
                din,
                false,
                !is_KC_not_CK,
                false,
                arch,
                stream_int,
                sac_shuffle_type,
                out_indices,
                tv::Tensor(),
                inp_indices,
                {AlgoHint.BackwardInput.value},
                1.0,
                beta);
            tv::Tensor a = out_bp;
            tv::Tensor b = features;
            tv::Tensor a_inds = out_indices;
            tv::Tensor b_inds = inp_indices;
            if (!is_KC_not_CK){{
                std::swap(a, b);
                std::swap(a_inds, b_inds);
            }}
            gemm_tuner.run_with_tuned_result(
                tuned_res_wgrad,
                a,
                b,
                dfilters.select(kv_dim, i),
                true,
                false,
                false,
                arch,
                stream_int,
                sab_shuffle_type,
                a_inds,
                b_inds,
                tv::Tensor(),
                {AlgoHint.BackwardWeight.value},
                1.0,
                beta);
            inited = true;
        }}
        """)
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def implicit_gemm(self):
        code = pccm.code()
        code.arg("allocator", "ExternalAllocator&")
        code.arg("conv_tuner", "ConvTuner&")
        code.arg("features, filters, pair_fwd", "tv::Tensor")
        code.arg("pair_mask_fwd_splits, mask_argsort_fwd_splits",
                 "std::vector<tv::Tensor>")
        code.arg("num_activate_out", "int")
        code.arg("masks", "tv::Tensor")
        code.arg("arch", "std::tuple<int, int>")

        code.arg("is_train, is_subm", "bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)",
                 "cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")
        code.arg("auto_fp32_accum", "bool", "true")
        code.arg("fp32_accum", "bool", "false")

        code.arg("bias", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("act_alpha", f"float", "0.0")
        code.arg("act_beta", f"float", "0.0")
        code.arg("act_type", f"tv::gemm::Activation", "tv::gemm::Activation::kNone", "cumm.tensorview.gemm.Activation = Activation.None_")
        code.arg("use_tf32", "bool", "true")
        code.arg("output_scale", "float", "1.0")
        code.arg("scale", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("output_add", "tv::Tensor", "tv::Tensor()",
                 "cumm.tensorview.Tensor = Tensor()")
        code.arg("output_add_scale", "float", "1.0")
        code.arg("output_dtype", "int", "-1")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code.ret("int")

        code.raw(f"""
        if (!bias.empty() || act_type != tv::gemm::Activation::kNone){{
            TV_ASSERT_RT_ERR(pair_mask_fwd_splits.size() == 1, "SplitGemm don't support fused bias/act for now.");
        }}
        uint32_t* mask_ptr = masks.data_ptr<uint32_t>();
        int num_mask = masks.dim(0);
        int out_channel = filters.dim(0);
        int in_channel = filters.dim(-1);
        int num_split = pair_mask_fwd_splits.size();
        TV_ASSERT_RT_ERR(num_mask == num_split, "error");
        filters = filters.view(out_channel, -1, in_channel);
        int kv = filters.dim(1);
        int mask_int_count = tv::div_up(kv, 32);
        tv::Tensor out_features;
        if (output_dtype < 0){{
            output_dtype = int(features.dtype());
        }}
        if (is_subm){{
            out_features = allocator.empty({pccm.literal(AllocKeys.OutFeatures)}, 
                {{num_activate_out, out_channel}}, tv::DType(output_dtype), features.device(), stream_int, false /*is_temp*/, output_scale);
        }}else{{
            out_features = allocator.zeros({pccm.literal(AllocKeys.OutFeatures)}, 
                {{num_activate_out, out_channel}}, tv::DType(output_dtype), features.device(), stream_int, false /*is_temp*/, output_scale);
        }}
        // auto start_ev = tv::CUDAEvent();
        // start_ev.record(stream_int);

        // auto arch = get_compute_capability();

        constexpr auto kForwardInt = static_cast<int>(tv::gemm::ConvOpType::kForward);
        constexpr auto kChannelLastInt = static_cast<int>(tv::gemm::ConvLayoutType::kChannelLast);
        auto tuned_res_exist = conv_tuner.get_tuned_algo(
            kForwardInt,
            int(features.dtype()),
            int(filters.dtype()),
            int(out_features.dtype()),
            out_channel, in_channel, arch);
        auto tune_res = std::get<0>(tuned_res_exist);
        auto exists = std::get<1>(tuned_res_exist);
        if (!exists){{
            auto tune_res_time = conv_tuner.tune_and_cache(
                kForwardInt,
                features, filters, out_features,
                kChannelLastInt,
                kChannelLastInt,
                kChannelLastInt,
                1, 1, 1, 
                arch,
                pair_mask_fwd_splits[0].type_view(tv::uint32),
                mask_argsort_fwd_splits[0],
                pair_fwd,
                false, // reverse_mask
                mask_ptr[0], // mask_filter
                -1,
                tv::Tensor(), // mask_output
                1.0, 0.0,
                stream_int, 
                auto_fp32_accum,
                fp32_accum,
                5, // num_run
                use_tf32,
                bias,
                scale);
            tune_res = std::get<0>(tune_res_time);
        }}
        float alpha = 1.0;
        if (tune_res.algo_desp.is_int8_inference){{
            alpha = output_scale;
        }}
        int mask_width = tune_res.algo_desp.tile_shape[0];
        tv::Tensor mask_output_fwd;
        std::vector<tv::Tensor> mask_output_fwd_splits;
        if (is_train){{
            mask_output_fwd = allocator.empty({pccm.literal(AllocKeys.MaskOutputFwd)}, 
                {{num_split, tv::div_up(num_activate_out, mask_width), mask_int_count}}, 
                tv::uint32, features.device(), stream_int);
            for (int i = 0; i < num_split; ++i){{
                mask_output_fwd_splits.push_back(mask_output_fwd[i]);
            }}
        }}else{{
            for (int i = 0; i < num_split; ++i){{
                mask_output_fwd_splits.push_back(tv::Tensor());
            }}
        }}
        
        for (int j = 0; j < num_split; ++j){{
            float beta = j == 0 ? 0 : 1;
            if (!bias.empty() && !tune_res.algo_desp.is_int8_inference){{
                // use source as bias
                beta = 1;
            }}
            if (!output_add.empty() && tune_res.algo_desp.is_int8_inference){{
                // use source as bias
                beta = output_add_scale / output_scale;
            }}

            if (j > 0){{
                bias = tv::Tensor();
            }}
            conv_tuner.run_with_tuned_result(
                tune_res,
                kForwardInt,
                features,
                filters,
                out_features,
                pair_mask_fwd_splits[j].type_view(tv::uint32),
                mask_argsort_fwd_splits[j],
                mask_output_fwd_splits[j],
                pair_fwd,
                false, // reverse_mask
                mask_ptr[j],
                -1, // mask_width
                alpha, beta,
                stream_int,
                tv::Tensor(), // workspace
                false, // verbose
                timer, 
                false,
                bias,
                act_alpha,
                act_beta,
                act_type,
                scale,
                output_add);
        }}
        // auto end_ev = tv::CUDAEvent();
        // end_ev.record(stream_int);
        // tv::ssprint(tune_res.algo_desp.__repr__(), "WTF", exists, 
        //     features.shape(), filters.shape(), out_features.shape(), tv::CUDAEvent::sync_and_duration(start_ev, end_ev));

        return std::make_tuple(mask_width, tune_res);
        """)
        return code.ret("std::tuple<int, ConvTuneResult>")

    @pccm.pybind.mark
    @pccm.static_function
    def implicit_gemm_backward(self):
        code = pccm.code()
        code.arg("allocator", "ExternalAllocator&")
        code.arg("conv_tuner", "ConvTuner&")
        code.arg("features, filters, out_bp, pair_fwd, pair_bwd", "tv::Tensor")
        code.arg("pair_mask_fwd_splits, pair_mask_bwd_splits",
                 "std::vector<tv::Tensor>")
        code.arg("mask_argsort_fwd_splits, mask_argsort_bwd_splits",
                 "std::vector<tv::Tensor>")
        code.arg("mask_output_fwd", "tv::Tensor")

        code.arg("masks", "tv::Tensor")
        code.arg("arch", "std::tuple<int, int>")

        code.arg("mask_width", "int")
        code.arg("is_subm", "bool")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)",
                 "cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")
        code.arg("auto_fp32_accum", "bool", "true")
        code.arg("fp32_accum", "bool", "false")
        code.arg("use_tf32", "bool", "true")

        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"TV_THROW_RT_ERR(\"not implemented for cpu!!!\")")
            return code

        code.raw(f"""

        auto filters_shape = filters.shape();
        auto filters_shape_vec = filters.shape_vector();

        uint32_t* mask_ptr = masks.data_ptr<uint32_t>();
        int num_mask = masks.dim(0);
        int out_channel = filters.dim(0);
        int in_channel = filters.dim(-1);
        int num_split = pair_mask_fwd_splits.size();
        TV_ASSERT_RT_ERR(num_mask == num_split, "error");
        filters = filters.view(out_channel, -1, in_channel);
        int kv = filters.dim(1);
        tv::Tensor din;
        if (is_subm){{
            din = allocator.empty({pccm.literal(AllocKeys.DIn)}, 
                features.shape_vector(), features.dtype(), features.device(), stream_int);
        }}else{{
            din = allocator.zeros({pccm.literal(AllocKeys.DIn)}, 
                features.shape_vector(), features.dtype(), features.device(), stream_int);
        }}
        tv::Tensor dfilters = allocator.zeros({pccm.literal(AllocKeys.DFilters)}, 
            filters_shape_vec, filters.dtype(), filters.device(), stream_int);
        dfilters = dfilters.view(out_channel, -1, in_channel);

        constexpr auto kForwardInt = static_cast<int>(tv::gemm::ConvOpType::kForward);
        constexpr auto kBackwardInputInt = static_cast<int>(tv::gemm::ConvOpType::kBackwardInput);
        constexpr auto kBackwardWeightInt = static_cast<int>(tv::gemm::ConvOpType::kBackwardWeight);

        constexpr auto kChannelLastInt = static_cast<int>(tv::gemm::ConvLayoutType::kChannelLast);

        // auto arch = get_compute_capability();

        auto dgrad_tuned_res_exist = conv_tuner.get_tuned_algo(
            kBackwardInputInt,
            int(din.dtype()),
            int(filters.dtype()),
            int(out_bp.dtype()),
            out_channel, in_channel, arch);
        auto wgrad_tuned_res_exist = conv_tuner.get_tuned_algo(
            kBackwardWeightInt,
            int(features.dtype()),
            int(dfilters.dtype()),
            int(out_bp.dtype()),
            out_channel, in_channel, arch, mask_width);

        auto dgrad_tune_res = std::get<0>(dgrad_tuned_res_exist);
        auto dgrad_exists = std::get<1>(dgrad_tuned_res_exist);
        auto wgrad_tune_res = std::get<0>(wgrad_tuned_res_exist);
        auto wgrad_exists = std::get<1>(wgrad_tuned_res_exist);

        if (!dgrad_exists){{
            tv::Tensor mask, mask_argsort;
            if (is_subm){{
                mask = pair_mask_fwd_splits[0].type_view(tv::uint32);
                mask_argsort = mask_argsort_fwd_splits[0];
            }}else{{
                mask = pair_mask_bwd_splits[0].type_view(tv::uint32);
                mask_argsort = mask_argsort_bwd_splits[0];
            }}
            auto tune_res_time = conv_tuner.tune_and_cache(
                kBackwardInputInt,
                din, filters, out_bp,
                kChannelLastInt,
                kChannelLastInt,
                kChannelLastInt,
                1, 1, 1, 
                arch,
                mask,
                mask_argsort,
                pair_bwd,
                is_subm, // reverse_mask
                mask_ptr[0], // mask_filter
                -1, // mask width
                tv::Tensor(), // mask_output
                1.0, 0.0,
                stream_int, 
                auto_fp32_accum,
                fp32_accum,
                5, // num_run
                use_tf32);
            dgrad_tune_res = std::get<0>(tune_res_time);
        }}
        if (!wgrad_exists){{
            auto tune_res_time = conv_tuner.tune_and_cache(
                kBackwardWeightInt,
                features, dfilters, out_bp,
                kChannelLastInt,
                kChannelLastInt,
                kChannelLastInt,
                1, 1, 1, 
                arch,
                mask_output_fwd[0].type_view(tv::uint32),
                mask_argsort_fwd_splits[0],
                pair_fwd,
                false, // reverse_mask
                mask_ptr[0], // mask_filter
                mask_width,
                tv::Tensor(), // mask_output
                1.0, 0.0,
                stream_int, 
                auto_fp32_accum,
                fp32_accum,
                5, // num_run
                use_tf32);
            wgrad_tune_res = std::get<0>(tune_res_time);
        }}
        int ws_size = conv_tuner.query_workspace_size(wgrad_tune_res.algo_desp,
                                               wgrad_tune_res.splitk,
                                               kBackwardWeightInt,
                                               pair_fwd.dim(1), in_channel,
                                               out_channel, kv);
        ExternalAllocator::guard_t workspace_guard;
        tv::Tensor workspace;
        if (ws_size > 0){{
            workspace_guard = allocator.empty_guard({{int64_t(ws_size)}}, tv::uint8, 0);
            workspace = workspace_guard->tensor;
        }}
        for (int j = 0; j < num_split; ++j){{
            tv::Tensor mask, mask_argsort;
            if (is_subm){{
                mask = pair_mask_fwd_splits[j].type_view(tv::uint32);
                mask_argsort = mask_argsort_fwd_splits[j];
            }}else{{
                mask = pair_mask_bwd_splits[j].type_view(tv::uint32);
                mask_argsort = mask_argsort_bwd_splits[j];
            }}
            float beta = j == 0 ? 0 : 1;
            conv_tuner.run_with_tuned_result(
                dgrad_tune_res,
                kBackwardInputInt,
                din,
                filters,
                out_bp,
                mask,
                mask_argsort,
                tv::Tensor(), // mask_output
                pair_bwd,
                is_subm, // reverse_mask
                mask_ptr[j],
                -1, // mask_width
                1.0, beta,
                stream_int,
                tv::Tensor(), // workspace
                false, // verbose
                timer);
            
            conv_tuner.run_with_tuned_result(
                wgrad_tune_res,
                kBackwardWeightInt,
                features, dfilters, out_bp,
                mask_output_fwd[j].type_view(tv::uint32),
                mask_argsort_fwd_splits[j],
                tv::Tensor(), // mask_output
                pair_fwd,
                false, // reverse_mask
                mask_ptr[j], // mask_filter
                mask_width,
                1.0, 0.0,
                stream_int, 
                workspace, // workspace
                false, // verbose
                timer);
        }}

        """)
        return code
