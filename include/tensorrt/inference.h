#include "NvInfer.h"
#include <memory>
#include <tensorview/tensor.h>
#include <unordered_map>
#include <vector>

namespace trt {

template <typename T> tv::DType trt_dtype_to_tv(T trt_dtype) {
  switch (trt_dtype) {
  case nvinfer1::DataType::kFLOAT:
    return tv::float32;
  case nvinfer1::DataType::kHALF:
    return tv::float16;
  case nvinfer1::DataType::kINT32:
    return tv::int32;
  case nvinfer1::DataType::kINT8:
    return tv::int8;
  default:;
  }
  TV_THROW_INVALID_ARG("unknown trt dtype");
}

struct InferDeleter {
  template <typename T> void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T> using trt_unique_ptr_t = std::unique_ptr<T, InferDeleter>;

class Logger : public nvinfer1::ILogger {
public:
  Logger(Severity severity = Severity::kWARNING)
      : reportableSeverity(severity) {}

  void log(Severity severity, const char *msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity)
      return;

    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      std::cerr << "INTERNAL_ERROR: ";
      break;
    case Severity::kERROR:
      std::cerr << "ERROR: ";
      break;
    case Severity::kWARNING:
      std::cerr << "WARNING: ";
      break;
    case Severity::kINFO:
      std::cerr << "INFO: ";
      break;
    default:
      std::cerr << "UNKNOWN: ";
      break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity;
};

class InferenceContext {
public:
  explicit InferenceContext(const std::string &engine_bin, int device)
      : logger_(nvinfer1::ILogger::Severity::kINFO), device_(device) {
    TV_ASSERT_INVALID_ARG(device >= 0, "invalid device id");
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (device >= deviceCount) {
      TV_THROW_INVALID_ARG("you provide device ", device, " but you only have ",
                           deviceCount, " device.");
    }
    cudaSetDevice(device);
    auto runtime = trt_unique_ptr_t<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_ =
        trt_unique_ptr_t<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(
            engine_bin.c_str(), engine_bin.size(), nullptr));
    ctx_ = trt_unique_ptr_t<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());

    max_batch_size_ = engine_->getMaxBatchSize();
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      auto dims = engine_->getBindingDimensions(i);
      std::vector<int> shape_vec(dims.d, dims.d + dims.nbDims);
      shape_vec.insert(shape_vec.begin(), {max_batch_size_});
      tv::TensorShape shape(shape_vec);
      std::string name = engine_->getBindingName(i);
      auto trt_dtype = engine_->getBindingDataType(i);
      auto tv_dtype = trt_dtype_to_tv(trt_dtype);
      bool isInput = engine_->bindingIsInput(i);
      name_to_idx_[name] = i;
      idx_to_name_[i] = name;
      name_to_host_mem_.insert({name, tv::Tensor(shape, tv_dtype, -1)});
      name_to_dev_mem_.insert({name, tv::Tensor(shape, tv_dtype, 0)});
      if (isInput)
        inp_idxes_.push_back(i);
      else
        out_idxes_.push_back(i);
      bindings_.push_back(name_to_dev_mem_[name].raw_data());
    }
    checkCudaErrors(cudaStreamCreate(&stream_));
  }

  std::unordered_map<std::string, tv::Tensor>
  operator()(std::vector<tv::Tensor> inputs) {
    TV_ASSERT_INVALID_ARG(inputs.size() == inp_idxes_.size(), "must provide",
                          inp_idxes_.size(), "inputs, but got", inputs.size());
    // inference batch size
    int bs = inputs[0].dim(0);
    for (auto &inp : inputs) {
      TV_ASSERT_INVALID_ARG(inp.dim(0) == bs,
                            "batch sizes of all input must same");
    }
    TV_ASSERT_INVALID_ARG(bs <= max_batch_size_, "your batchsize too large", bs,
                          max_batch_size_);
    for (int i = 0; i < inputs.size(); ++i) {
      auto &dev_mem = name_to_dev_mem_[idx_to_name_[i]];
      auto shape_inp = inputs[i].shape().subshape(1);
      auto shape_dev = dev_mem.shape().subshape(1);
      TV_ASSERT_INVALID_ARG(shape_inp == shape_dev,
                            "shape except batch must same", shape_inp,
                            shape_dev);
      dev_mem.slice_first_axis(0, bs).copy_(inputs[i].slice_first_axis(0, bs),
                                            stream_);
    }

    ctx_->enqueue(bs, bindings_.data(), stream_, nullptr);

    for (int i : out_idxes_) {
      name_to_host_mem_[idx_to_name_[i]].slice_first_axis(0, bs).copy_(
          name_to_dev_mem_[idx_to_name_[i]].slice_first_axis(0, bs), stream_);
    }
    checkCudaErrors(cudaStreamSynchronize(stream_));
    std::unordered_map<std::string, tv::Tensor> output_map;
    for (int i = 0; i < out_idxes_.size(); ++i) {
      auto name = idx_to_name_[out_idxes_[i]];
      output_map[name] = name_to_host_mem_[name].slice_first_axis(0, bs);
    }
    return output_map;
  }

  std::unordered_map<std::string, tv::Tensor>
  operator()(std::unordered_map<std::string, tv::Tensor> inputs) {
    std::vector<tv::Tensor> inputs_vec(inp_idxes_.size());
    int count = 0;
    for (auto &p : inputs) {
      auto iter = name_to_idx_.find(p.first);
      TV_ASSERT_INVALID_ARG(iter != name_to_idx_.end(), "cant find your name",
                            p.first);
      inputs_vec[name_to_idx_[p.first]] = p.second;
    }
    TV_ASSERT_INVALID_ARG(count == inp_idxes_.size(), "your inp not enough");
    return (*this)(inputs_vec);
  }

  tv::Tensor operator[](std::string name) {
    auto iter = name_to_host_mem_.find(name);
    if (iter == name_to_host_mem_.end()) {
      TV_THROW_INVALID_ARG(name, "not found.");
    }
    return iter->second;
  }

  std::string repr() {
    std::stringstream ss;
    ss << "InferenceContext[gpu=" << device_ << "]";
    ss << "\n  Inputs:";
    std::string name;
    for (auto &i : inp_idxes_) {
      name = idx_to_name_[i];
      auto &mem = name_to_host_mem_[name];
      ss << "\n    " << name << "[" << tv::detail::typeString(mem.dtype())
         << "]: " << mem.shape();
    }
    ss << "\n  Outputs:";
    for (auto &i : out_idxes_) {
      name = idx_to_name_[i];
      auto &mem = name_to_host_mem_[name];
      ss << "\n    " << name << "[" << tv::detail::typeString(mem.dtype())
         << "]: " << mem.shape();
    }
    return ss.str();
  }

private:
  Logger logger_;
  trt_unique_ptr_t<nvinfer1::ICudaEngine> engine_;
  trt_unique_ptr_t<nvinfer1::IExecutionContext> ctx_;
  std::unordered_map<std::string, tv::Tensor> name_to_dev_mem_;
  std::unordered_map<std::string, tv::Tensor> name_to_host_mem_;
  std::unordered_map<std::string, int> name_to_idx_;
  std::unordered_map<int, std::string> idx_to_name_;
  std::vector<int> inp_idxes_;
  std::vector<int> out_idxes_;
  std::vector<void *> bindings_;
  cudaStream_t stream_;
  int max_batch_size_;
  int device_;
};

} // namespace trt
