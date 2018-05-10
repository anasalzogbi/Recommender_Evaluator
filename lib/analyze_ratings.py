"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - analyzing the users activity and items popularity in a given ratings file
 - Reducing the ratings matrix by applying a constraint for minimum activity/popularity ans saving the result
Date: October 28th, 2017
"""

import sys
import os
import argparse
import csv
import numpy as np
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.files_utils import read_ratings_into_array, read_docs_vocabs_file, save_mult_list, load_list_from_file, save_list


def apply_threshold(ratings_array, u_threshold, p_threshold):
    i = 0
    r_sum_1 = np.sum(ratings_array, axis=1)
    r_sum_0 = np.sum(ratings_array, axis=0)
    min_pop = min(r_sum_0)
    min_act = min(r_sum_1)

    u_ids_stack = []
    p_ids_stack = []

    while min_pop < p_threshold or min_act < u_threshold:

        # Applying the min_ratings constraint
        u_ids = np.where(r_sum_1 >= u_threshold)
        ratings_array = ratings_array[u_ids]
        if min(ratings_array.shape) == 0:
            print("Ratings array don't comply with the constraints!")
            return None
        u_ids_stack.append(dict(enumerate(u_ids[0])))
        i += 1

        r_sum_0 = np.sum(ratings_array, axis=0)
        min_pop = min(r_sum_0)
        if min_pop >= p_threshold:
            print("Done after {} rounds".format(i))
            break

        # Applying the min_popularity constraint
        p_ids = np.where(r_sum_0 >= p_threshold)
        ratings_array = (ratings_array.T[p_ids]).T
        if min(ratings_array.shape) == 0:
            print("Ratings array don't comply with the constraints!")
            return None
        p_ids_stack.append(dict(enumerate(p_ids[0])))
        i += 1

        r_sum_1 = np.sum(ratings_array, axis=1)
        min_act = min(r_sum_1)
        if min_act >= u_threshold:
            print("Done after {} rounds".format(i))
            break
        if i % 10 == 0:
            print("Iteration {}, (min_u_ratings, min_p_pop) = ({},{})".format(i, min_act, min_pop))
    u_ids = range(ratings_array.shape[0])
    p_ids = range(ratings_array.shape[1])

    while len(u_ids_stack) != 0:
        l = []
        d = u_ids_stack.pop()
        for u_id in u_ids:
            l.append(d[u_id])
        u_ids = l[:]

    while len(p_ids_stack) != 0:
        l = []
        d = p_ids_stack.pop()
        for p_id in p_ids:
            l.append(d[p_id])
        p_ids = l[:]
    return ratings_array, u_ids, p_ids


def get_ratings_analysis(ratings):
    r_sum_1 = np.sum(ratings, axis=1)
    r_sum_0 = np.sum(ratings, axis=0)
    out = "Ratings matrix: {}".format(ratings.shape) + "\n"
    out += "# ratings: {}".format(int(ratings.sum())) + "\n"
    out += "# min_activity, min_popularity: {}, {}".format(int(min(r_sum_1)), int(min(r_sum_0))) + "\n"
    out += "# max_activity, max_popularity: {}, {}".format(int(max(r_sum_1)), int(max(r_sum_0)))
    return out


if __name__ == '__main__':
    ratings_file = "../../citeulike-crawled/converted_data/2k_1_P3/terms_keywords_based/users.dat"
    min_act_threshold = 100
    min_popularity_threshold = 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_file", "-f", help="The users' ratings file")
    parser.add_argument("--min_activity_threshold", "-a", type=int, help="The minimum users' ratings.")
    parser.add_argument("--min_popularity_threshold", "-p", type=int, help="The minimum papers' popularity.")

    args = parser.parse_args()
    if args.ratings_file:
        ratings_file = args.ratings_file
    if not os.path.exists(ratings_file):
        print("Error: ratings file doen' exist: {}".format(ratings_file))
        raise ValueError
    if args.min_activity_threshold:
        min_act_threshold = args.min_activity_threshold
    if args.min_popularity_threshold:
        min_popularity_threshold = args.min_popularity_threshold

    t0 = time.time()
    ratings = read_ratings_into_array(ratings_file)
    t1 = time.time()
    print("Ratings read in {:5.2f} minutes".format((t1 - t0) / 60))
    print(get_ratings_analysis(ratings))
    t2 = time.time()
    print("Ratings printed in {:5.2f} minutes".format((t2 - t1) / 60))
    res = apply_threshold(ratings, min_act_threshold, min_popularity_threshold)
    t3 = time.time()
    print("Ratings reduced in {:5.2f} minutes".format((t3 - t2) / 60))

    if res:
        reduced_ratings, u_ids, p_ids = res
        stats = get_ratings_analysis(reduced_ratings)
        print(stats)
        # Read users mappings:
        userhash_map = os.path.join(os.path.dirname(ratings_file), "citeulikeUserHash_userId_map.dat")
        M = {}
        with open(userhash_map) as f:
            reader = csv.reader(f, delimiter=" ")
            next(reader, None)  # skip the headers
            for row in reader:
                M[int(row[1])] = row[0]

        output_folder = os.path.join(os.path.dirname(ratings_file),
                                     "2k_{}_P{}_reduced".format(min_act_threshold, min_popularity_threshold))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate the new_user_id -> user_hash mapping:
        print("Generating the new_user_id mappings...")
        new_userhash_map = [(M[j], i) for i, j in enumerate(u_ids)]
        new_userhash_map_file = os.path.join(output_folder, "userId_citeulikeUserHash_map.dat".format(min_act_threshold,
                                                                                                      min_popularity_threshold))
        with open(new_userhash_map_file, 'w') as user_map_out:
            user_map_out.write(" ".join(["user_id", "user_citeulike_hash"]) + "\n")
            for (uh, uid) in new_userhash_map:
                # Write the user id mapping:
                user_map_out.write(" ".join([str(uid), uh]) + "\n")
        # Read papers mappings:
        print("Generating the papers mapping...")
        citeulikeid_map = os.path.join(os.path.dirname(ratings_file), "citeulikeId_docId_map.dat")
        M = {}
        with open(citeulikeid_map) as f:
            reader = csv.reader(f, delimiter=" ")
            next(reader, None)  # skip the headers
            for row in reader:
                M[int(row[1])] = row[0]

        # Generate the new_doc_id -> citeulike_id mapping:
        new_citeulikeid_map = [(M[j], i) for i, j in enumerate(p_ids)]
        new_citeulikeid_map_file = os.path.join(output_folder, "docId_citeulikeId_map.dat".format(min_act_threshold,
                                                                                                  min_popularity_threshold))
        with open(new_citeulikeid_map_file, 'w') as paper_map_out:
            paper_map_out.write(" ".join(["doc_id", "citeulike_id"]) + "\n")
            for (pcid, pid) in new_citeulikeid_map:
                # Write the user id mapping:
                paper_map_out.write(" ".join([str(pid), str(pcid)]) + "\n")

        # Generating the new mult.dat file
        print("Generating the new mult file...")
        # 1- load the mult file:
        # papers = read_docs_vocabs_file(os.path.join(os.path.dirname(ratings_file),"mult.dat"))
        docs_vocabs = read_docs_vocabs_file(os.path.join(os.path.dirname(ratings_file), "mult.dat"))
        # 2- Get the term ids of the remaining papers (column ids which cells are nonzero):
        # term_ids = np.sum(papers[p_ids], axis = 0).nonzero()[0]
        term_ids = set()
        for p in p_ids:
            for e in docs_vocabs[p]:
                term_ids.add(e[0])
        term_ids = sorted(list(term_ids))
        t_dict = dict(zip(term_ids, range(len(term_ids))))

        # 3- Generate new mult (reduced)
        new_papers = []
        for p in p_ids:
            new_papers.append([(t_dict[t], tf) for (t, tf) in docs_vocabs[p]])

        """
        new_papers = []
        for p in p_ids:
            ids = papers[p, term_ids].nonzero()[0]
            new_papers.append([(i, papers[p, term_ids][i]) for i in ids])
        """
        # 4- Save the new mult:
        save_mult_list(new_papers, os.path.join(output_folder, "mult.dat"))

        # 5- Generate the new terms file:
        # Load terms:
        terms = load_list_from_file(os.path.join(os.path.dirname(ratings_file), "terms.dat"))

        # Filter the terms:
        terms = terms[term_ids]

        # 4- Save the new terms:
        save_list(terms, os.path.join(output_folder, "terms.dat"))

        print("Saving the reduced ratings matrix...")
        # Saving the reduced ratings matrix in user.dat
        new_ratings_file = os.path.join(output_folder, "users.dat")
        with open(new_ratings_file, 'w') as ratings_out:
            for u_ratings in reduced_ratings:
                ratings_out.write(
                    " ".join(map(lambda x: str(x), [int(sum(u_ratings))] + list(np.where(u_ratings > 0)[0]))) + "\n")
        # Save stats:
        with open(os.path.join(output_folder, "stats.txt"), 'w') as f:
            f.write(stats)

            # np.save(os.path.join(os.path.dirname(ratings_file), "users_reduced_act_{}_pop_{}".format(min_act_threshold, min_popularity_threshold)), reduced_ratings)
