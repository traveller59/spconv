# Frequently Asked Questions

- [What does no dependency on pytorch mean?](#What-does-no-dependency-on-pytorch-mean)

## What does no dependency on pytorch mean?

This means spconv 2.x doesn't have pytorch shared library dependency when you use ```ldd``` to inspect required shared objects of our shared library.

This **doesn't** mean spconv 2.x can run in pytorch with **any** version.

Most of pytorch extension repos use official pytorch extension build system, libraries built from these extension depend on pytorch c++ library and impossible to match requirements of [manylinux](https://github.com/pypa/manylinux). The official python package server, [PyPI](https://pypi.org/), and its mirrors, only accept manylinux package for linux platforms. So we must remove all pytorch stuffs from our c++ code to create manylinux packages.

Spconv 2.x use two core feature of pytorch to match manylinux requirements: ```torch.Tensor.data_ptr``` and ```torch.cuda.current_stream().cuda_stream```. the first one is used to get pointer of ```torch.Tensor```, the second part is used to get cuda stream pointer. We don't need pytorch anymore in c++ code when these features are available in pytorch.