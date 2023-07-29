from typing import Tuple

import torch

class TensorStack:
    def __init__(self, batch_size: int, data_shape: Tuple[int], init_size=int(1e2), dtype=torch.float32, use_resize_ = False) -> None:
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.init_size = init_size
        self.size = self.init_size
        self.dtype = dtype
        self.use_resize_ = use_resize_
        
        # Holds the data
        self.storage = torch.zeros((self.batch_size, self.size, ) + self.data_shape, dtype=self.dtype)
        # Points to the head
        self.heads = torch.full((self.batch_size,), -1, dtype=torch.int64)
        # Index telling which batch each input is from
        self.index = torch.arange(self.batch_size, dtype=torch.int64)
        
        # Dummy mask for batch
        self.dummy_batch_mask = torch.ones((self.batch_size,), dtype=torch.bool)
        
    def resize_storage(self):
        self.size *= 2
        if self.use_resize_:
            self.storage.resize_((self.batch_size, self.size, ) + self.data_shape)
        else:
            self.storage = torch.cat([self.storage, torch.zeros_like(self.storage)], dim=1)
    
    def push(self, data: torch.Tensor, batch_mask: torch.BoolTensor = None):
        if batch_mask is None:
            batch_mask = self.dummy_batch_mask
        
        self.heads[batch_mask] += 1
        selected_heads = self.heads[batch_mask]
        if torch.any(selected_heads == self.size):
            self.resize_storage()
        self.storage[batch_mask, selected_heads] = data
        
    def pop(self, batch_mask: torch.BoolTensor = None) -> torch.Tensor:
        if batch_mask is None:
            batch_mask = self.dummy_batch_mask

        selected_heads = self.heads[batch_mask]
        if torch.any(selected_heads == -1):
            raise ValueError(f"Tried to pop from {torch.sum((self.heads == -1).long())} stacks that were empty.")
        result_data = self.storage[batch_mask, selected_heads]
        result_index = self.index[batch_mask]
        self.heads[batch_mask] -= 1
        
        return result_data, result_index
        
    def empty(self) -> torch.BoolTensor:
        return (self.heads == -1)
    
    @property
    def counts(self) -> torch.LongTensor:
        return self.heads + 1
    
    def drop_batches(self, drop_mask: torch.BoolTensor):
        self.storage = self.storage[~drop_mask]
        self.heads = self.heads[~drop_mask]
        self.index = self.index[~drop_mask]
        self.dummy_batch_mask = self.dummy_batch_mask[~drop_mask]
        self.batch_size = self.storage.shape[0]
    
    def get_data(self) -> torch.Tensor:
        return self.storage[:, :torch.max(self.counts)]

class SimplePQ:
    def __init__(self, batch_size: int, data_shape: Tuple[int], init_size=int(1e2), dtype=torch.float32, use_resize_ = False) -> None:
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.init_size = init_size
        self.size = self.init_size
        self.dtype = dtype
        self.use_resize_ = use_resize_
        
        # Holds the data
        self.storage = torch.zeros((self.batch_size, self.size, ) + self.data_shape, dtype=self.dtype)
        # Holds the keys - these are probably what will require 
        self.keys = torch.full((self.batch_size, self.size), -float("inf"))
        # Index telling which batch each input is from
        self.index = torch.arange(self.batch_size, dtype=torch.int64)
        # Count telling how many entries each batch has
        self.counts = torch.zeros(self.batch_size, dtype=torch.int64)
        # What is the next location that is free
        self.free_stack = TensorStack(self.batch_size, data_shape=(), dtype=torch.int64, use_resize_=True)
        self.free_stack.push(torch.zeros(self.batch_size, dtype=torch.int64))
        
        # Dummy mask for batch
        self.dummy_batch_mask = torch.ones((self.batch_size,), dtype=torch.bool)
    
    def pop_free(self, batch_mask: torch.BoolTensor) -> torch.LongTensor:
        next_location, _ = self.free_stack.pop(batch_mask)
        # Preserve the inductive hypothesis that last entry of free stack
        # is always the next index of unused space
        is_stack_empty = self.free_stack.empty()
        # Index out the empty stacks in the batch mask. The new mask is where we are empty and in the batch mask
        self.free_stack.push(next_location[is_stack_empty[batch_mask]] + 1, (is_stack_empty & batch_mask))
        
        return next_location
    
    def resize_storage(self):
        self.size *= 2
        self.storage = torch.cat([self.storage, torch.zeros_like(self.storage)], dim=1)
        self.keys = torch.cat([self.keys, torch.full_like(self.keys, fill_value=-float("inf"))], dim=1)
    
    def insert(self, keys: torch.FloatTensor, data: torch.Tensor, batch_mask: torch.BoolTensor = None):
        """
        Args:
            keys (float): The key value for sorting the data
            data (`self.data_type`): The data to insert
            batch_mask (bool, Optional): Which batches to insert into

        Shape:
            - keys: (sum(batch_mask),)
            - data: (sum(batch_mask), *self.data_shape)
            - batch_mask: (batch_size,)
            
        Modifies:
            Will insert data and keys into next free spot. If none available
            will double storage size.
        """
        if batch_mask is None:
            batch_mask = self.dummy_batch_mask
        
        insert_index = self.pop_free(batch_mask)
        if torch.any(insert_index >= self.size):
            self.resize_storage()
        
        self.keys[batch_mask, insert_index] = keys
        self.storage[batch_mask, insert_index] = data
        self.counts[batch_mask] += 1
    
    def pop(self, batch_mask: torch.BoolTensor = None) -> Tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]:
        if batch_mask is None:
            batch_mask = self.dummy_batch_mask
        
        pop_inds = torch.argmax(self.keys[batch_mask], dim=1)
        self.free_stack.push(pop_inds, batch_mask)
                
        result_keys = self.keys[batch_mask, pop_inds]
        result_data = self.storage[batch_mask, pop_inds]
        result_index = self.index[batch_mask]
        
        self.keys[batch_mask, pop_inds] = -float("inf")
        self.counts[batch_mask] -= 1
        return (result_keys, result_data, result_index)
        
    def drop_batches(self, drop_mask: torch.BoolTensor):
        self.storage = self.storage[~drop_mask]
        self.keys = self.keys[~drop_mask]
        self.index = self.index[~drop_mask]
        self.dummy_batch_mask = self.dummy_batch_mask[~drop_mask]
        self.counts = self.counts[~drop_mask]
        self.free_stack.drop_batches(drop_mask)
        self.batch_size = self.storage.shape[0]
        
        
        
                