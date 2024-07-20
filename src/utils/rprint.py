# coding: utf-8

"""
custom print and log functions 
"""

__all__ = ['rprint', 'rlog']

def rprint(*args, **kwargs):
    # Remove unsupported keyword arguments
    unsupported_kwargs = ['style', 'duration']
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_kwargs}
    print(*args, **filtered_kwargs)

def rlog(*args, **kwargs):
    # Remove unsupported keyword arguments
    unsupported_kwargs = ['style', 'duration']
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_kwargs}
    print(*args, **filtered_kwargs)


