#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>

#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>

#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>

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

int main(){
    tv::ssprint("Hello libspconv!!!");

    return 0;
}