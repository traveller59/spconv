# TensorRT Int8 Guide


## Prerequisites

### Plugin

Due to limitation of tensorrt, following requirements must be satisfied:

1. pad all inputs to a static shape

2. use a tensor to save current number of voxels, copy it to cpu and slice all inputs to real shape during inference (enqueue).

3. ```supportsFormatCombination``` must allow exactly one combination, i.e. we must set dtype of this layer during network build. for example, if we want to use fp16, this function must accept fp16 and reject other dtypes to avoid tensorrt perform dtype/format selection during engine build.

4. Number of dimensions of int8 tensor for plugin must larger or equal to 3. (tested in tensorrt 8.4)

5. TensorRT version >= 8.4, tensorrt 8.0 don't support int8 plugin


### Pytorch

* PTQ/QAT model is ready

### Spconv Int8 Scale/Bias Format

basic rule:

```C++
fp32_data = float(int8_data) * scale
int8_data = int8_t(saturate(round(fp32_data / scale)))
```

assume we have a pytorch quantized layer, the required scale/bias in spconv int8 is:

```Python
import spconv.pytorch.quantization.quantized.reference as snnqr
import spconv.pytorch.quantization.intrinsic.quantized as snniq
import spconv.pytorch.quantization.quantized as snnq

input_scale = ...
output_scale = ...
if isinstance(layer, snnqr.SpConv):
    q_weight = layer.get_quantized_weight() # for snnqr.SpConv
    bias_np = layer.bias.detach().cpu().numpy()
elif isinstance(layer, (snniq.SparseConvReLU, snniq.SparseConvAddReLU, snnq.SparseConv))
    q_weight = layer.weight() # for quantized layers
    bias_np = layer.bias().detach().cpu().numpy()
else:
    raise NotImplementedError
w_perchannel_scales = q_weight.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)

scale_for_spconv_implicit_gemm = (input_scale * w_perchannel_scales) / output_scale
bias_for_spconv_implicit_gemm = bias_np / output_scale

```

then we can feed them to ```implicit_gemm```:
```C++
// output_add and output_add_scale: for fused conv-add-relu layer
ConvGemmOps::implicit_gemm(
    allocator, tuner, features_int8, weight_int8, pair_fwd,
    pair_mask_splits, mask_argsort_splits, actual_out_feature_num,
    mask_tensor, arch, false, is_subm,
    reinterpret_cast<std::uintptr_t>(stream), tv::CUDAKernelTimer(false),
    false, false, bias_for_spconv_implicit_gemm, 1.0,
    0.0, tv::gemm::Activation::kReLU, false /*use_tf32*/, output_scale, 
    scale_for_spconv_implicit_gemm, output_add, output_add_scale);
```

### Explicit Mode or Implicit

There are two int8 mode in tensorrt: implicit and explicit.

For Implicit, we can use tensorrt int8 calibrator to calculate scale and use them in plugin. This isn't tested and doesn't covered here.

For Explicit, we insert qdq to network, tensorrt will fuse QDQ and convert layers to quantized based on QDQ layers. see [this doc](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks).

There is a important drawback in tensorrt int8: tensorrt won't fuse QDQ for custom int8 plugins. So we must fuse QDQ by ourself (in pytorch), and **keep QDQ** in regular layers such linear and convolution.

Pytorch will add QDQ in ```convert_fx``` and ```convert_to_reference_fx```.

```convert_to_reference_fx```: insert qdq and convert fused module to reference, but it **doesn't** fuse any QDQ in your network. If we don't want to write fuse code manually, we can't use this function.

```convert_fx```: insert qdq and convert fused module to quantized for native (CPU) backend. this function will fuse **ALL** QDQs in your network, if we want to use tensorrt explicit quantization, we must keep QDQ for regular layers.

Currently we implement this via pytorch ```convert_fx``` and use some hack:

```Python
import torch.ao.nn.intrinsic as nni
import torch.nn.quantized._reference as nnqr

from torch.ao.quantization.fx._lower_to_native_backend import \
    STATIC_LOWER_FUSED_MODULE_MAP, STATIC_LOWER_MODULE_MAP, QBIN_OP_MAPPING
from spconv.pytorch.quantization.backend_cfg import \
    SPCONV_STATIC_LOWER_FUSED_MODULE_MAP, SPCONV_STATIC_LOWER_MODULE_MAP
# add spconv layers to support QDQ fusion for sparse conv layers
STATIC_LOWER_FUSED_MODULE_MAP.update(SPCONV_STATIC_LOWER_FUSED_MODULE_MAP)
STATIC_LOWER_MODULE_MAP.update(SPCONV_STATIC_LOWER_MODULE_MAP)
# remove linear layers to avoid QDQ fusion for linear.
STATIC_LOWER_FUSED_MODULE_MAP.pop(nni.LinearReLU)
STATIC_LOWER_MODULE_MAP.pop(nnqr.Linear)
# run above BEFORE convert_fx
converted_model = qfx.convert_fx(prepared_model, qconfig_mapping=qconfig_mapping, backend_config=backend_cfg)
# or just use spconvq.prepare_spconv_torch_inference(True)
```
We can also use a spconv function to handle this if regular layers in your network only contains Linear:
```Python 
spconvq.prepare_spconv_torch_inference(with_linear=False)
```

If your network contains convolutions, you can do same thing for conv layers. this isn't covered in ```spconvq.prepare_spconv_torch_inference```.



## Steps 

### Record number of voxels for each layer

there is a argument in ```SparseConvolution``` layers: ```record_voxel_count```. If you enable it, max number of voxels will be recorded in a registered buffer during inference. Turn on it and run inference in whole training dataset.

After inference, we know max number of voxels of each spconv layer, which is required in tensorrt plugin.

### write ```torch.fx``` based torch->trt conversion

After PTQ/QAT model ready, we can use [```torch.fx.Interpreter```](https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter) to transform traced pytorch model to tensorrt.

see [example](../example/mnist/mnist_net_transform.py).




