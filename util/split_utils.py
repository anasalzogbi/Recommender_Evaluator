"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Several functionalities for split functions
Date: December 2017
"""
import random


def random_divide(lst, k):
    """
    Randomly splits the items of a given list into k lists
    :param lst: the input list
    :param k: the number of resulting lists
    :return: 2d list
    """
    res = []
    random.shuffle(lst)
    partition_size = len(lst) // k
    for fold in range(k):
        if fold == k - 1:
            res.append(lst[fold * partition_size:len(lst)])
        else:
            res.append(lst[fold * partition_size:fold * partition_size + partition_size])
    return res

def random_sample_list(lst, n):
    """
    Randomly samples n items from the list lst
    """
    if n > len(lst):
        n = len(lst)
    random.shuffle(lst)
    return lst[:n]

def format_grid(ax):
    """
    For plotting: Formats lines in a given axes. 
    :param ax: The axes
    """
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines()

    for line in ticklines:
        line.set_linewidth(3)

    for line in gridlines:
        line.set_linestyle('-.')

    for label in ax.get_yticklabels():
        label.set_color('r')
        label.set_fontsize(12)
    for label in ax.get_xticklabels():
        label.set_color('r')
        label.set_fontsize('small')