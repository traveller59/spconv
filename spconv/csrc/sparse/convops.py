import pccm

from cumm.gemm.main import GemmMainUnitTest
from cumm.conv.main import ConvMainUnitTest
from .alloc import ExternalAllocator
from spconv.core import ConvAlgo
from spconv.constants import SpconvAllocatorKeys
from cumm.constants import CUMM_CPU_ONLY_BUILD
from cumm.common import GemmBasicHost, TensorView, NlohmannJson

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
        code.raw(f"return splitk > 0 && std::get<0>(arch) > 0")
        return code

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
        code.ctor_init("algo_desp", "tv::gemm::ConvAlgoDesp()")
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
        code.raw(f"return splitk > 0 && std::get<0>(arch) > 0")
        return code

class GemmTunerSimple(pccm.ParameterizedClass):
    def __init__(self, gemm_cu: GemmMainUnitTest, conv_cu: ConvMainUnitTest):
        super().__init__()
        self.add_dependency(ExternalAllocator, GemmTuneResult,
                            ConvTuneResult, TensorView)
        self.add_param_class("gemm", gemm_cu, "GemmMain")
        self.add_param_class("conv", conv_cu, "ConvMain")
        self.add_include("tensorview/utility/tuplehash.h")

        self.add_member("desps_", "std::vector<tv::gemm::GemmAlgoDesp>")
        self.add_member("nvrtc_progs_", "std::unordered_map<std::string, tv::NVRTCProgram>")
        self.add_member("nvrtc_caches_", "std::unordered_map<std::tuple<std::string, int, int, std::uintptr_t>, tv::NVRTCModule>")

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("desps", "std::vector<tv::gemm::GemmAlgoDesp>")
        code.arg("nvrtc_progs", "std::unordered_map<std::string, std::string>")
        code.ctor_init("desps_", "desps")
        code.raw(f"""
        for (auto& v : nvrtc_progs){{
            const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(v.second.c_str());
            nvrtc_progs_.insert(v.first, tv::NVRTCProgram::from_binary(code_ptr, v.second.size()));
        }}
        """)
        return code 

    @pccm.member_function
    def get_all_available(self):
        code = pccm.code()
        code.arg("a, b, c", "tv::Tensor")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.arg("nvrtc_progs", "std::unordered_map<std::string, std::string>")
        code.ctor_init("desps_", "desps")
        code.raw(f"""
        
        for (auto& v : nvrtc_progs){{
            const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(v.second.c_str());
            nvrtc_progs_.insert(v.first, tv::NVRTCProgram::from_binary(code_ptr, v.second.size()));
        }}

        """)
        return code 



class ConvGemmOps(pccm.ParameterizedClass):

    def __init__(self, gemm_cu: GemmMainUnitTest, conv_cu: ConvMainUnitTest):
        super().__init__()
        self.add_dependency(ExternalAllocator, GemmTuneResult,
                            ConvTuneResult)
        self.add_param_class("gemm", gemm_cu, "GemmMain")
        self.add_param_class("conv", conv_cu, "ConvMain")

    @pccm.pybind.mark
    @pccm.static_function
    def indice_conv(self):
        """1. this function need to take a out features
        that from subm first mm.
        2. this function don't support CPU.
        """
        code = pccm.code()
        code.arg("allocator", "ExternalAllocator&")
        code.arg("out_features_after_mm", "tv::Tensor")
        code.arg("features, filters, indice_pairs", "tv::Tensor")
        code.arg("indice_pair_num", "tv::Tensor")
        code.arg("num_activate_out", "int")
        code.arg("inverse", "bool", "false")
        code.arg("subm", "bool", "false")
        code.arg("algo", "int", f"{ConvAlgo.Native.value}")
        code.arg("filter_hwio", "bool", "false")
        code.arg("stream_int", f"std::uintptr_t", "0", pyanno="int")
        if CUMM_CPU_ONLY_BUILD:
            code.raw(f"""
            throw std::runtime_error("this function can only be used with CUDA.")
            """)
            return code.ret("tv::Tensor")

        code.raw(f"""
        TV_ASSERT_RT_ERR(!features.is_cpu(), "this function don't support cpu.")
        int out_channel;
        if (filter_hwio){{
            out_channel = filters.dim(-1);
        }}else{{
            out_channel = filters.dim(-2);
        }}
        filters = filters.view(-1, filters.dim(-2), filters.dim(-1));
        int kv = filters.dim(0);
        int kv_center = kv / 2;
        tv::Tensor out_features;
        if (kv == 1 && subm){{
            return;
        }}
        auto indice_pair_num_cpu = indice_pair_num.cpu();
        auto indice_pair_num_cpu_ptr = indice_pair_num_cpu.data_ptr<int>();
        int maxnhot = 0;
        bool all_zero = true;
        for (int i = 0; i < kv; ++i){{
            if (indice_pair_num_cpu_ptr[i] != 0){{
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
        
        """)

        return code
