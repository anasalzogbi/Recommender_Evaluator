"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Read/ Write rating matrices from/to csv files
Date: October 22nd, 2017
"""

import csv
import os
import numpy as np


def read_mappings(mappings_file, delimiter=","):
    """
    reads mappings from a given file into a dictionary: 1st column (key) -> 2nd column (value)

    :param mappings_file: The input file
    :param delimiter: The file delimiter
    :return: dictionary: key (1st column) -> value (2nd column) 
    """
    dic = {}
    with open(mappings_file) as f:
        # Skip the header:
        f.readline()
        for line in f:
            id1, id2 = line.split(delimiter)
            dic[id1.strip()] = int(id2)
    return dic


def read_ratings(ratings_file, delimiter=" ", lda_c_format=True):
    """
    reads the ratings file into 2d list
    :param ratings_file: the path of the ratings_list file, the file is expected to be formated as following: 
    line i has space separated list of item ids rated by user i, the first value is the number of rated items.
    :param delimiter:
    :return: 2d list, the ith row contains a list of relevant items ids to user i
    """
    start = 0

    # lda-c format: first value of each row is the row length
    if lda_c_format:
        start = 1

    ratings_list = []
    with open(ratings_file) as f:
        for line in f:
            ratings = [int(x) for x in line.replace("\n", "").split(delimiter)[start:] if x != ""]
            ratings_list.append(ratings)
    return ratings_list


def read_ratings_into_array(ratings_file):
    """
    Reads the ratings file into a numpy 2d array, doesn' require users and items count
    :param ratings_file: the path of the ratings file, the file is expected to be formated as following: 
    The first entry in line i is the ratings count (n) of user (i), the rest n-length space separated list contains the
    item ids rated by user i. 
    :return: numpy 2d array, the ith row contains a list of relevant items ids to user i
    """
    items_set = set()
    l = []
    with open(ratings_file) as f:
        user_id = 0
        for line in f:
            ratings = [int(x) for x in line.replace("\n", "").split(" ")[1:] if x != ""]
            l.append(ratings)
            for r in ratings:
                items_set.add(r)
            user_id += 1

    print("#Users: {}, #Items: {}".format(user_id, len(items_set)))
    ratings_mat = np.zeros((user_id, len(items_set)))
    for u_id, u_list in enumerate(l):
        for i in u_list:
            ratings_mat[u_id, i] = 1
    return ratings_mat


def write_ratings(ratings_list, filename, delimiter=" ", print_line_length=True):
    """
    writes user matrix to a file, the file will be formated as following: line i has delimiter-separated list of item ids rated by user i
    :param ratings_list: users 2d list, row num = num_users
    :param filename: the path of the users file
    :param delimiter: default: space
    :param print_line_length: if True: the first column of each line will record the line's length
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for ratings in ratings_list:
            if print_line_length:
                writer.writerow([len(ratings)] + ratings)
            else:
                writer.writerow(ratings)


def write_list_to_file(input_list, output_folder, delimiter=" ", header=None):
    """
    Writes a list into a file
    :param input_list: list of items, which elements will be written
    :param output_folder: the output file
    :param delimiter: the delimiter for the output file
    :param header: the output file header
    """
    with open(output_folder, 'w') as doc_out:
        if header:
            doc_out.write(delimiter.join(header) + "\n")
        for element in input_list:
            doc_out.write(delimiter.join([str(i) for i in element]) + "\n")


def print_list(lst):
    print('[%s]' % ', '.join(map(str, lst)))


def read_docs_vocabs_file(doc_vocab_file, with_header=False, lda_c_format=True, delimiter=' '):
    start = 0

    # lda-c format: first value of each row is the #vocab
    if lda_c_format:
        start = 1
    doc_vocab = []
    vocabs = set()
    with open(doc_vocab_file, 'r') as f:
        line_num = 0
        if with_header:
            next(f)
        for line in f:
            doc_vector = []
            line_split = line.strip().split(delimiter)
            for e in line_split[start:]:
                vocab, freq = e.split(":")
                try:
                    vocab = int(vocab)
                    doc_vector.append((vocab, int(freq)))
                    vocabs.add(vocab)
                except ValueError:
                    print("Error in doc_vocab file {} : line {}, value {} is not int.".format(doc_vocab_file, line_num,
                                                                                              e))
                    raise
            doc_vocab.append(doc_vector)
            line_num += 1
    if (len(vocabs) != max(vocabs) + 1):
        print("Error in doc_vocab file {}: # Vocabs = {}, max vocab = {}".format(doc_vocab_file, len(vocabs),
                                                                                 max(vocabs)))
        raise ValueError
    return doc_vocab
    """
    mat = np.zeros((len(doc_vocab), len(vocabs)))
    doc_idx = 0
    for d in doc_vocab:
        for e in d:
            mat[doc_idx, e[0]] = e[1]
            if e[1] <= 0:
                print(e[1])
        doc_idx += 1
    return mat
    """


def save_mult_list(arr, output_file, delimiter=" "):
    with open(output_file, 'w') as f:
        for row in arr:
            f.write(str(len(row)) + delimiter + delimiter.join([str(i) + ":" + str(j) for i, j in row]) + "\n")


def load_list_from_file(input_file):
    l = []
    with open(input_file) as f:
        for line in f:
            l.append(line.strip())
    return np.array(l)


def save_list(arr, output_file):
    with open(output_file, 'w') as f:
        for row in arr:
            f.write(row + "\n")
