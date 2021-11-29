from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class HashTable:
    key_itemsize: int
    value_itemsize: int
    is_cpu: bool
    insert_count: int
    def __init__(self, is_cpu: bool, key_itemsize: int, value_itemsize: int, keys_data: Tensor, values_data: Tensor, stream: int = 0) -> None: 
        """
        Args:
            is_cpu: 
            key_itemsize: 
            value_itemsize: 
            keys_data: 
            values_data: 
            stream: 
        """
        ...
    def clear(self, stream: int = 0) -> None: 
        """
        in this function, if values is empty, it will be assigned to zero.
                
        Args:
            stream: 
        """
        ...
    def insert(self, keys: Tensor, values: Tensor =  Tensor(), stream: int = 0) -> None: 
        """
        in this function, if values is empty, it will be assigned to zero.
                
        Args:
            keys: 
            values: 
            stream: 
        """
        ...
    def query(self, keys: Tensor, values: Tensor, is_empty: Tensor, stream: int) -> None: 
        """
        query keys, save to values, and save is_empty to is_empty
                
        Args:
            keys: 
            values: 
            is_empty: 
            stream: 
        """
        ...
    def assign_arange_(self, count: Tensor, stream: int = 0) -> None: 
        """
        this function assign "arange(NumItem)" to table values.
        useful in "unique-like" operations.
        unlike insert/query, this method only support i32/i64/u32/u64 for value.
        count must be u32/u64.
        Args:
            count: 
            stream: 
        """
        ...
    def size_cpu(self) -> int: 
        """
        this function can only be used to get cpu hash table size.
                
        """
        ...
    def items(self, keys: Tensor, values: Tensor, count: Tensor, stream: int) -> None: 
        """
        get items.
                
        Args:
            keys: 
            values: 
            count: 
            stream: 
        """
        ...
    def insert_exist_keys(self, keys: Tensor, values: Tensor, is_empty: Tensor, stream: int) -> None: 
        """
        insert v of given k if k exists. won't insert any new key.
                
        Args:
            keys: 
            values: 
            is_empty: 
            stream: 
        """
        ...
