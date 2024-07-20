# coding: utf-8

"""
custom print and log functions 
"""

__all__ = ['rprint', 'rlog']

def rprint(*args, **kwargs):
    kwargs.pop('style', None)  # Remove 'style' if it is present
    print(*args, **kwargs)

def rlog(*args, **kwargs):
    kwargs.pop('style', None)  # Remove 'style' if it is present
    print(*args, **kwargs)

