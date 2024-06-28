# ==============================================================================#
# Authors: Windsor Nguyen
# File: nearest_power_of_2.py
#
# ==============================================================================#


def nearest_power_of_2(x: int):
    s = bin(x)
    s = s.lstrip('-0b')
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length
