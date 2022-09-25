import spconv 


from spconv.core_cc.cumm.common import CompileInfo
if __name__ == "__main__":
    print(CompileInfo.arch_is_compatible_gemm((9, 0)), CompileInfo.arch_is_compiled_gemm((9, 0)))
    print(CompileInfo.arch_is_compatible_gemm((8, 6)), CompileInfo.arch_is_compiled_gemm((8, 6)))