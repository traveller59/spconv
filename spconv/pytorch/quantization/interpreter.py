from typing import Any, Dict, List, Optional, Set, Type

import torch
import torch.fx

REGISTERED_NODE_HANDLERS: Dict[Any, Any] = {}


def register_node_handler(*names):

    def wrap_func(handler):
        global REGISTERED_NODE_HANDLERS
        for n in names:
            REGISTERED_NODE_HANDLERS[n] = handler

        def new_handler(inputs, attributes, scope):
            return handler(inputs, attributes, scope)

        return new_handler

    return wrap_func


def register_method_handler(name: str, tensor_classes):
    if not isinstance(tensor_classes, (list, tuple)):
        tensor_classes = [tensor_classes]

    def wrap_func(handler):
        global REGISTERED_NODE_HANDLERS
        for tcls in tensor_classes:
            REGISTERED_NODE_HANDLERS[(tcls, name)] = handler

        def new_handler(inputs, attributes, scope):
            return handler(inputs, attributes, scope)

        return new_handler

    return wrap_func


def get_node_handler(name):
    global REGISTERED_NODE_HANDLERS
    msg = "missing handler " + str(name)
    msg += ", available handlers: {}".format(
        list(REGISTERED_NODE_HANDLERS.keys()))
    assert name in REGISTERED_NODE_HANDLERS, msg
    return REGISTERED_NODE_HANDLERS[name]


class NetworkInterpreter(torch.fx.Interpreter):

    def __init__(self,
                 network_ctx,
                 module: torch.fx.GraphModule,
                 inputs: List[Any],
                 verbose: bool = False):
        super().__init__(module)
        self.network_ctx = network_ctx
        self._inputs = inputs
        self._outputs = None
        self._cur_node_name: Optional[str] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._verbose = verbose

    def run(self):
        super().run(*self._inputs)
        assert self._outputs is not None
        return self._outputs

    def run_node(self, n):
        self._cur_node_name = str(n)
        return super().run_node(n)

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        type_str = submod_type.__qualname__
        type_str_parts = type_str.split(".")
        msg = f"[Module.{type_str_parts[-1]}]{target}({args}|{kwargs}) => "

        try:
            converter = get_node_handler(submod_type)
            res = converter(self.network_ctx, submod, args, kwargs,
                            self._cur_node_name)
            msg += f"{res}"
            if self._verbose:
                print(msg)
            return res
        except Exception as e:
            if self._verbose:
                print(msg)
            raise e

    def call_function(self, target, args, kwargs):
        msg = f"[Func]{target}({args}|{kwargs}) => "
        try:
            converter = get_node_handler(target)
            res = converter(self.network_ctx, target, args, kwargs,
                            self._cur_node_name)
            msg += f"{res}"
            if self._verbose:
                print(msg)
            return res
        except Exception as e:
            if self._verbose:
                print(msg)
            raise e

    def call_method(self, target, args, kwargs):
        msg = f"[Method]{target}({args}|{kwargs}) => "
        assert isinstance(target, str)
        try:
            key = (type(args[0]), target)
            converter = get_node_handler(key)
            res = converter(self.network_ctx, target, args, kwargs,
                            self._cur_node_name)
            msg += f"{res}"
            if self._verbose:
                print(msg)
            return res
        except Exception as e:
            if self._verbose:
                print(msg)
            raise e

    def output(self, target, args, kwargs):
        self._outputs = args
