from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
class CompileInfo:
    @staticmethod
    def get_compiled_cuda_arch() -> List[Tuple[int, int]]: ...
