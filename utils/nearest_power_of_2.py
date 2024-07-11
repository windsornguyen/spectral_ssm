# =============================================================================#
# Authors: Windsor Nguyen
# File: nearest_power_of_two.py
# =============================================================================#

"""
Utility functions to compute the nearest power of two.

Adapted from the bit_length stdlib function:
https://docs.python.org/3/library/stdtypes.html
"""

def nearest_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than or equal to x.
    If x is already a power of 2, it returns x itself.
    Otherwise, it returns the next higher power of 2.

    Args:
        x (int): The input integer.

    Returns:
        int: The smallest power of 2 that is greater than or equal to x.
    """
    s = bin(x)
    s = s.lstrip("-0b")
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length
