# coding: utf-8

"""
custom print and log functions 
"""

__all__ = ['rprint', 'rlog']

def rprint(*args, **kwargs):
    # Remove unsupported keyword arguments
    unsupported_kwargs = ['style', 'duration']
    for kwarg in unsupported_kwargs:
        kwargs.pop(kwarg, None)
    print(*args, **kwargs)

def rlog(*args, **kwargs):
    # Remove unsupported keyword arguments
    unsupported_kwargs = ['style', 'duration']
    for kwarg in unsupported_kwargs:
        kwargs.pop(kwarg, None)
    print(*args, **kwargs)

