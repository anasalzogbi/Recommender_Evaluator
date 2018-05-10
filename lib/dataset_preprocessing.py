"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Converting data structure (document-vocab files and ratings file) from id-based into line-based structure
Date: October 25, 2017
"""
import argparse
import os
import shutil


# Convert document-vocab file from id-based structure into line-based structure
def convert_doc_vocab_file(doc_vocab_file, doc_vocab_file_out, doc_id_map_file_out, with_header=True, id_separator=',', delimiter=' ', out_delimiter=" "):
    doc_id_map = {}
    with open(doc_vocab_file, 'r') as f, open(doc_vocab_file_out, 'w') as doc_vocab_out, \
            open(doc_id_map_file_out, 'w') as doc_id_out:
        doc_id_out.write(out_delimiter.join(["citeulike_id", "doc_id"]) + "\n")
        line_num = 0
        if with_header:
            next(f)
        for line in f:
            # Read from doc_vocab file:
            citeulike_id, vocabs_vector = line.strip().split(id_separator)

            # Write the document id mapping:
            doc_id_out.write(out_delimiter.join([citeulike_id, str(line_num)]) + "\n")
            doc_id_map[citeulike_id] = str(line_num)

            # write the doc_vocab:
            vocab_ids = vocabs_vector.split(delimiter)
            doc_vocab_out.write(out_delimiter.join([str(len(vocab_ids))] + vocab_ids) + "\n")

            # increase line_numer:
            line_num += 1
    return doc_id_map


# Convert ratings file from id-based structure into line-based structure
def convert_ratings_file(ratings_file, ratings_file_out, user_id_map_file_out, doc_id_map, with_header=False, id_separator=';', delimiter=',', out_delimiter=" "):
    user_id_map = {}
    with open(ratings_file, 'r') as f, open(ratings_file_out, 'w') as ratings_out, open(user_id_map_file_out, 'w') as user_map_out:
        user_map_out.write(out_delimiter.join(["user_citeulike_hash", "user_id"]) + "\n")
        line_num = 0
        if with_header:
            next(f)
        for line in f:
            # Read from ratings file:
            user_hash, docs_vector = line.strip().split(id_separator)

            # pick up the ratings from the dictionary:
            docs_ids = docs_vector.split(delimiter)
            r = [doc_id_map[doc_id] for doc_id in docs_ids if doc_id in doc_id_map]

            if len(r) > 0:
                # Write the user id mapping:
                user_map_out.write(out_delimiter.join([user_hash, str(line_num)]) + "\n")

                # write the ratings:
                ratings_out.write(out_delimiter.join([str(len(r))] + r) + "\n")

                # increase line_numer:
                line_num += 1


# main code;
if __name__ == '__main__':
    input_dir = '../../citeulike-crawled/2k_1_P3'
    output_directory = '../../citeulike-crawled/converted_data/2k_1_P3'

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", help="The output directory")
    parser.add_argument("--input_dir", "-i", help="The directory which contains input files")

    args = parser.parse_args()

    if args.output_dir:
        output_directory = args.output_dir

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.input_dir:
        input_dir = args.input_dir

    # A- terms:
    t_output_directory = os.path.join(output_directory, "terms_based")
    if not os.path.exists(t_output_directory):
        os.makedirs(t_output_directory)
    doc_vocab_file = os.path.join(input_dir, 'document-terms.txt')
    if not os.path.exists(doc_vocab_file):
        print("Error: input document-vocabulary file not exists: {}".format(doc_vocab_file))
        raise ValueError("Error: input document-vocabulary file not exists.")

    ratings_file = os.path.join(input_dir, 'userLibraries.txt')
    if not os.path.exists(ratings_file):
        print("Error: input ratings file not exists: {}".format(ratings_file))
        raise ValueError("Error: input ratings file not exists.")

    doc_vocab_file_out = os.path.join(t_output_directory, "mult.dat")
    doc_id_map_file_out = os.path.join(t_output_directory, "citeulikeId_docId_map.dat")

    # 1- Convert doc_vocab file:
    doc_id_map = convert_doc_vocab_file(doc_vocab_file, doc_vocab_file_out, doc_id_map_file_out, with_header=False)

    ratings_file_out = os.path.join(t_output_directory, "users.dat")
    user_id_map_file_out = os.path.join(t_output_directory, "citeulikeUserHash_userId_map.dat")
    # 2- Convert ratings file:
    convert_ratings_file(ratings_file, ratings_file_out, user_id_map_file_out, doc_id_map)

    # 3- copy the vocab file:
    shutil.copy2(os.path.join(input_dir, 'terms.dat'), os.path.join(t_output_directory, "terms.dat") )

    # B-Keywords:
    k_output_directory = os.path.join(output_directory, "keywords_based")
    if not os.path.exists(k_output_directory):
        os.makedirs(k_output_directory)

    doc_vocab_file = os.path.join(input_dir, 'document-keywords.txt')
    if not os.path.exists(doc_vocab_file):
        print("Error: input document-keywords file not exists: {}".format(doc_vocab_file))
        raise ValueError("Error: input document-keywords file not exists.")

    doc_vocab_file_out = os.path.join(k_output_directory, "mult.dat")
    doc_id_map_file_out = os.path.join(k_output_directory, "citeulikeId_docId_map.dat")

    # 1- Convert doc_vocab file:
    doc_id_map = convert_doc_vocab_file(doc_vocab_file, doc_vocab_file_out, doc_id_map_file_out, with_header=False)

    ratings_file_out = os.path.join(k_output_directory, "users.dat")
    user_id_map_file_out = os.path.join(k_output_directory, "citeulikeUserHash_userId_map.dat")
    # 2- Convert ratings file:
    convert_ratings_file(ratings_file, ratings_file_out, user_id_map_file_out, doc_id_map)

    # 3- copy the vocab file:
    shutil.copy2(os.path.join(input_dir, 'keywords.dat'), os.path.join(k_output_directory, "terms.dat"))

    # C-Terms_Keywords:
    tk_output_directory = os.path.join(output_directory,"terms_keywords_based")
    if not os.path.exists(tk_output_directory):
        os.makedirs(tk_output_directory)

    doc_vocab_file = os.path.join(input_dir, 'document-terms_and_keywords.txt')
    if not os.path.exists(doc_vocab_file):
        print("Error: input document-terms_and_keywords file not exists: {}".format(doc_vocab_file))
        raise ValueError("Error: input document-terms_and_keywords file not exists.")

    doc_vocab_file_out = os.path.join(tk_output_directory, "mult.dat")
    doc_id_map_file_out = os.path.join(tk_output_directory, "citeulikeId_docId_map.dat")

    # 1- Convert doc_vocab file:
    doc_id_map = convert_doc_vocab_file(doc_vocab_file, doc_vocab_file_out, doc_id_map_file_out, with_header=False)

    ratings_file_out = os.path.join(tk_output_directory, "users.dat")
    user_id_map_file_out = os.path.join(tk_output_directory, "citeulikeUserHash_userId_map.dat")
    # 2- Convert ratings file:
    convert_ratings_file(ratings_file, ratings_file_out, user_id_map_file_out, doc_id_map)
    
    # 3- copy the vocab file:
    shutil.copy2(os.path.join(input_dir, 'terms_and_keywords.dat'), os.path.join(tk_output_directory, "terms.dat"))