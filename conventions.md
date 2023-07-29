# Definining Classes

## Inside the __init__ method
1. When saving input arguments as attributes of the object, use the attributes for all future references (e.g. call `self._____` instead of `_____` within `__init__`).

# Tensors
1. For tensors that only hold binary values, use `torch.CharTensor` (`torch.int8`) instead of `torch.BoolTensor` or `torch.ByteTensor`. Both are stored as the same underlying data. Use `torch.BoolTensor` when we want to semantically distinguish the tensor from just a numerical tensor. The use of `torch.CharTensor` is so we can represent binary numbers as well as signed numbers
   - This is a new addition as of 2022-12-14. Older code uses `torch.BoolTensor` or `torch.ByteTensor` incorrectly instead of `torch.CharTensor`.
   - Also we may be able to change most uses of `torch.LongTensor` to `torch.CharTensor`.
2. Use `.shape` instead of `.size()`. Both return the same object, but `.shape` is cross-compatible with other similar libraries like `numpy` and `tensorflow`.
   - This is a new addition as of 2022-12-14. Older code may use `.size()`.

# Defining Functions
1. Use `dim` or `dims` (as opposed to `axis` and `axes`) as a keyword argument for functions that work with torch tensors because this is the standard within the torch library. 

# Channel Coding Design
1. Modulation should be done separately from encoder ??? TODO: any continuous encoder is doing "modulation" together with encoding.

# Code Documentation
We'll use numpy's documentation format. See `example_numpy.py` for examples of how the docstrings should be organized. We don't need to fill out all fields (parameters, notes, returns, etc.), only what is useful for development and code comprehension.
1. For tensors with known shapes write 
   ```
   - tensor_name (s1 x s2 x s3) : description...
   ```

# Chunking
I'll view the chunk_size as the maximum number of bytes I want to store in memory at once. Note that this is not how it was viewed before 2023-01-07