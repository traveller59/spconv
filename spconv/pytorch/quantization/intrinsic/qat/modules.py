# torch.ao.nn.intrinsic.qat.modules.conv_fused
import math
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import TypeVar
from spconv.pytorch.conv import SparseConvolution
from typing import List, Optional, Tuple, Union
from spconv.core import ConvAlgo
from cumm import tensorview as tv
from spconv.pytorch.core import SparseConvTensor
import spconv.pytorch.quantization.intrinsic as snni
from spconv.pytorch.quantization.utils import fuse_spconv_bn_weights
MOD = TypeVar('MOD', bound=SparseConvolution)

class _SparseConv(SparseConvolution):

    _FLOAT_MODULE = MOD
    _FLOAT_CONV_MODULE = SparseConvolution

    def __init__(self,
                 ndim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 subm: bool = False,
                 output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 transposed: bool = False,
                 inverse: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                 act_alpha: float = 0,
                 act_beta: float = 0,
                 name=None,
                 qconfig=None,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        SparseConvolution.__init__(self, ndim, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
            bias=False,
            subm=subm,
            output_padding=output_padding,
            transposed=transposed,
            inverse=inverse,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            record_voxel_count=record_voxel_count,
            act_type=act_type,
            act_alpha=act_alpha,
            act_beta=act_beta,
            name=name, **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return self._conv_forward(self.training, input, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        assert issubclass(type(mod), cls._FLOAT_MODULE), (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
            + f" not {type(mod).__qualname__}"
        )
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if issubclass(type(mod), nni._FusedModule):
            mod = mod[0]  # type: ignore[index]
        conv: SparseConvolution = mod
        qconfig = mod.qconfig
        qat_conv = cls(conv.ndim, conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, 
                         conv.bias is not None,
                         subm=conv.subm,
                         output_padding=conv.output_padding,
                         transposed=conv.transposed,
                         inverse=conv.inverse,
                         indice_key=conv.indice_key,
                         algo=conv.algo,
                         fp32_accum=conv.fp32_accum,
                         record_voxel_count=conv.record_voxel_count,
                         act_type=conv.act_type,
                         act_alpha=conv.act_alpha,
                         act_beta=conv.act_beta,
                         name=conv.name,
                         qconfig=qconfig)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        """ This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.ndim,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            subm=self.subm,
            output_padding=self.output_padding,
            transposed=self.transposed,
            inverse=self.inverse,
            indice_key=self.indice_key,
            algo=self.algo,
            fp32_accum=self.fp32_accum,
            record_voxel_count=self.record_voxel_count,
            act_type=self.act_type,
            act_alpha=self.act_alpha,
            act_beta=self.act_beta,
            name=self.name)
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        # conv relu
        if issubclass(cls, nni._FusedModule):
            modules = [conv]
            assert hasattr(cls, "_FLOAT_RELU_MODULE")
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            fused = cls._FLOAT_MODULE(*modules)  # type: ignore[arg-type, attr-defined, operator]
            fused.train(self.training)
            return fused
        else:
            return conv

class SparseConv(_SparseConv, SparseConvolution):
    r"""
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as :class:`~torch.nn.Conv1d`

    Similar to :class:`~torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = SparseConvolution
    _FLOAT_CONV_MODULE = SparseConvolution

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)

class SparseConvReLU(SparseConv, nni._FusedModule):
    r"""A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = snni.SpconvReLUNd
    _FLOAT_CONV_MODULE = SparseConvolution
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def forward(self, input):
        x = self._conv_forward(self.training, input, self.weight_fake_quant(self.weight), self.bias)
        return x.replace_feature(F.relu(x.features))

    @classmethod
    def from_float(cls, mod):
        return super(SparseConvReLU, cls).from_float(mod)

class SparseConvAddReLU(SparseConv, nni._FusedModule):
    r"""A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = snni.SpconvAddReLUNd
    _FLOAT_CONV_MODULE = SparseConvolution
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def forward(self, input, add_input):
        x = self._conv_forward(self.training, input, self.weight_fake_quant(self.weight), self.bias,
            add_input=add_input)
        return x.replace_feature(F.relu(x.features))

    @classmethod
    def from_float(cls, mod):
        return super(SparseConvAddReLU, cls).from_float(mod)


class _SparseConvBn(SparseConvolution, nni._FusedModule):

    _version = 2
    _FLOAT_MODULE = MOD
    _FLOAT_CONV_MODULE = SparseConvolution

    def __init__(self,
                 # SparseConvolution args
                 ndim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 subm: bool = False,
                 output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 transposed: bool = False,
                 inverse: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                 act_alpha: float = 0,
                 act_beta: float = 0,
                 name=None,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        SparseConvolution.__init__(self, ndim, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
            bias=False,
            subm=subm,
            output_padding=output_padding,
            transposed=transposed,
            inverse=inverse,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            record_voxel_count=record_voxel_count,
            act_type=act_type,
            act_alpha=act_alpha,
            act_beta=act_beta,
            name=name)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        self._enable_slow_path_for_better_numerical_stability = False

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_SparseConvBn, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _forward(self, input: SparseConvTensor, add_input: Optional[SparseConvTensor] = None):
        assert not self._enable_slow_path_for_better_numerical_stability
        if self._enable_slow_path_for_better_numerical_stability:
            return self._forward_slow(input)
        return self._forward_approximate(input, add_input)

    def _forward_approximate(self, input: SparseConvTensor, add_input: Optional[SparseConvTensor] = None):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.features.dtype)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.features.dtype)
        conv_spt = self._conv_forward(self.training, input, scaled_weight, zero_bias)
        conv = conv_spt.features
        conv_orig = conv / scale_factor# .reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias# .reshape(bias_shape)
        conv = self.bn(conv_orig)
        if add_input is not None:
            conv = conv + add_input.features
        conv_spt = conv_spt.replace_feature(conv)
        return conv_spt

    def _forward_slow(self, input: SparseConvTensor):
        """
        TODO not implemented for now
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """

        assert self.bn.running_var is not None
        assert self.bn.running_mean is not None

        # using zero bias here since the bias for original conv
        # will be added later
        zero_bias = torch.zeros(self.out_channels, device=self.weight.device, dtype=input.features.dtype)

        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        conv_out = torch.Tensor()
        if self.bn.training:
            # needed to compute batch mean/std
            conv_spt = self._conv_forward(self.training, input, self.weight, zero_bias)
            conv_out = conv_spt.features
            # update bn statistics
            with torch.no_grad():
                conv_out_bias = (
                    conv_out if self.bias is None else conv_out + self.bias.reshape(bias_shape)
                )
                self.bn(conv_out_bias)

        # fused conv + bn without bias using bn running statistics
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.weight_fake_quant(
            self.weight * scale_factor.reshape(weight_shape)
        )
        # fused conv without bias for inference: (r * W / running_std) * X
        conv_bn_spt = self._conv_forward(self.training, input, scaled_weight, zero_bias)
        conv_bn = conv_bn_spt.features
        if self.bn.training:
            avg_dims = [0] + list(range(2, len(self.weight.shape)))
            batch_mean = conv_out.mean(avg_dims)
            batch_var = torch.square(conv_out - batch_mean.reshape(bias_shape)).mean(
                avg_dims
            )
            batch_std = torch.sqrt(batch_var + self.bn.eps)

            # scale to use batch std in training mode
            # conv(X, r * W / std_Y) = conv(X, r * W / running_std) * (running_std / std_Y)
            unscale_factor = running_std / batch_std
            conv_bn *= unscale_factor.reshape(bias_shape)

            fused_mean = batch_mean
            fused_std = batch_std
        else:
            fused_mean = self.bn.running_mean - (self.bias if self.bias is not None else 0)
            fused_std = running_std

        # fused bias = beta - r * mean / std
        fused_bias = self.bn.bias - self.bn.weight * fused_mean / fused_std
        conv_bn += fused_bias.reshape(bias_shape)

        # HACK to let conv bias particpiate in loss to avoid DDP error (parameters
        #   were not used in producing loss)
        if self.bias is not None:
            conv_bn += (self.bias - self.bias).reshape(bias_shape)
        conv_bn_spt = conv_bn_spt.replace_feature(conv_bn)
        return conv_bn_spt

        return conv_bn

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_SparseConvBn, self).extra_repr()

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_SparseConvBn, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        # The ignore is because _FLOAT_MODULE is a TypeVar here where the bound
        # has no __name__ (code is fine though)
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        conv: SparseConvolution = mod[0]
        bn: nn.BatchNorm1d = mod[1]
        qat_convbn = cls(conv.ndim, conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, 
                         conv.bias is not None,
                         subm=conv.subm,
                         output_padding=conv.output_padding,
                         transposed=conv.transposed,
                         inverse=conv.inverse,
                         indice_key=conv.indice_key,
                         algo=conv.algo,
                         fp32_accum=conv.fp32_accum,
                         record_voxel_count=conv.record_voxel_count,
                         act_type=conv.act_type,
                         act_alpha=conv.act_alpha,
                         act_beta=conv.act_beta,
                         name=conv.name,
                         eps=bn.eps, momentum=bn.momentum,
                         freeze_bn=False,
                         qconfig=qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.ndim,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            subm=self.subm,
            output_padding=self.output_padding,
            transposed=self.transposed,
            inverse=self.inverse,
            indice_key=self.indice_key,
            algo=self.algo,
            fp32_accum=self.fp32_accum,
            record_voxel_count=self.record_voxel_count,
            act_type=self.act_type,
            act_alpha=self.act_alpha,
            act_beta=self.act_beta,
            name=self.name)
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        if cls._FLOAT_BN_MODULE:  # type: ignore[attr-defined]
            # fuse bn into conv
            conv.weight, conv.bias = fuse_spconv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias
            )
        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv

class SparseConvBn(_SparseConvBn):
    r"""
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    # base class defines _FLOAT_MODULE as "ConvBn1d"
    _FLOAT_MODULE = snni.SpconvBnNd  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = SparseConvolution
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = None
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = snni.SpconvReLUNd

class SparseConvBnReLU(_SparseConvBn):
    r"""
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    # base class defines _FLOAT_MODULE as "ConvBn1d"
    _FLOAT_MODULE = snni.SpconvBnReLUNd  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = SparseConvolution
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = snni.SpconvReLUNd

    def forward(self, input):
        x = _SparseConvBn._forward(self, input)
        return x.replace_feature(F.relu(x.features))

    @classmethod
    def from_float(cls, mod):
        return super(SparseConvBnReLU, cls).from_float(mod)

class SparseConvBnAddReLU(_SparseConvBn):
    r"""
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    # base class defines _FLOAT_MODULE as "ConvBn1d"
    _FLOAT_MODULE = snni.SpconvBnAddReLUNd  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = SparseConvolution
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = snni.SpconvAddReLUNd

    def forward(self, input, add_input):
        x = _SparseConvBn._forward(self, input, add_input)
        return x.replace_feature(F.relu(x.features))

    @classmethod
    def from_float(cls, mod):
        return super(SparseConvBnAddReLU, cls).from_float(mod)
