"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Splitting a given ratings matrix into training, test and validation sets following different split schemes
Date: October 22nd, 2017
"""
import sys
import os
import numpy as np
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.files_utils import read_ratings
from util.files_utils import write_ratings
from util.split_utils import random_divide
from util.stats import Stats

class Range(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


class Splitter(object):
    def __init__(self, root_path, split_method="in-matrix-item", generate_validation=False, folds_num = 5, delimiter=" "):
        self.users_ratings = read_ratings(os.path.join(root_path, "users.dat"))
        self.root_path = root_path
        self.num_users = len(self.users_ratings)
        self.delimiter = delimiter
        self.split_method = split_method
        self.num_items = max([max(i) for i in self.users_ratings]) + 1
        self.out_folder = os.path.join(self.root_path, self.split_method + "_folds")
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        # List for split stats, the list is written to a file in the split() method:
        self.stats_list = []
        stats_header = ['{:20}'.format("Stats_of"), '{:4}'.format('fold'), '{:4}'.format('min'), '{:4}'.format('max'), '{:6}'.format('avg'), '{:3}'.format('std')]
        self.stats_list.append(stats_header)
        self.generate_validation = generate_validation
        self.folds_num = folds_num

    def items_mat_from_users_ratings(self, users_ratings):
        """
        construct item matrix from user matrix
        :return: 2d list, row num = num_items, the ith row contains all user ids who rated item i
        """
        items_mat = [[] for _ in range(self.num_items)]
        num_users = len(users_ratings)
        for user in range(num_users):
            item_ids = users_ratings[user]
            for item in item_ids:
                try:
                    items_mat[item].append(user)
                except IndexError:
                    print(len(items_mat), item, user, self.num_items, self.num_users)
                    raise IndexError
        for item in range(self.num_items):
            items_mat[item].sort()
            return items_mat

    def users_mat_from_items(self, items):
        """
        Construct user matrix from item matrix
        :param items: 2d list, row num = num_item, the ith row contains all user ids who rated item i
        :return: 2d list, row num = num_user, the ith row contains item ids which rated by user i
        """
        users = [[] for _ in range(self.num_users)]
        num_items = len(items)
        for i in range(num_items):
            user_ids = items[i]
            for user in user_ids:
                users[user].append(i)
        for user in range(self.num_users):
            users[user].sort()
        return users


    def create_all_folds_test_split_matrix(self, folds_num=5):
        """
        Creates the splits matrix for test data, the result after invoking this method is a
        single file that saves an ndarray of shape:(num_users, num_folds, list of test ids).
        The list of test ids contains both user_positive_ids and user_fold_unrated_items
        :param folds_num: the number of folds, default 5
        :return: None
        """
        num_users = len(self.users_ratings)
        print("Number of users: {}".format(self.num_users))
        print("Creating all folds testsplits, progress:")
        splits = [[[] for _ in range(folds_num)] for _ in range(num_users)]
        items_ids = set(range(self.num_items))
        for user in range(num_users):
            user_items = self.users_ratings[user]
            unrated_items = list(items_ids - set(user_items))
            splits[user] = random_divide(unrated_items, folds_num)
            if user % 500 == 0:
                print("user_{}".format(user))
        for fold in range(folds_num):
            print("Calculating fold_{}".format(fold + 1))
            rated_items_test_fold = read_ratings(
                os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-users.dat".format(fold + 1)))
            for user in range(num_users):
                splits[user][fold].extend(rated_items_test_fold[user])
                splits[user][fold].sort()
        splits = np.array(splits)

        # Calculate statistics:
        for fold in range(folds_num):
            users_test_stats = [len(i) for i in splits[:, fold]]
            self.stats_list.append(['{:20}'.format("item_in_test_per_usr"), '{:4d}'.format(fold + 1),
                                    '{:4d}'.format(min(users_test_stats)),
                                    '{:4d}'.format(max(users_test_stats)), '{:6.1f}'.format(np.mean(users_test_stats)),
                                    '{:3.1f}'.format(np.std(users_test_stats))])

        print("Saving the splits ndarray to: {}".format("splits.npy"))
        np.save(os.path.join(self.out_folder, "splits"), splits)

    def cf_split(self, folds_num=5):
        """
        Splits the rating matrix following the in-matrix method defined in CTR, the result after invoking this method is:
        two files for each fold (cf-train-fold_id-users.dat and cf-train-fold_id-users.dat), both files have the same following format:
        line i has delimiter-separated list of item ids rated by user i
        :param folds_num: the number of folds, default 5
        :return: None
        """
        items_mat = self.items_mat_from_users_ratings(self.users_ratings)
        train = [[[] for _ in range(self.num_items)] for _ in range(folds_num)]
        test = [[[] for _ in range(self.num_items)] for _ in range(folds_num)]
        validation = [[[] for _ in range(self.num_items)] for _ in range(folds_num)]
        print("Number of items: {}".format(self.num_items))
        folds_list = list(range(folds_num))
        print("Splitting items ratings, progress:")

        # 1- Split items ratings into the folds. This guarantees that all items appear at least once in the test set.
        # If generating validation set is required:
        if self.generate_validation:
            for item in range(self.num_items):
                # Reporting progress:
                if item % 5000 == 0:
                    print("doc_{}".format(item))

                user_ids = np.array(items_mat[item])
                n = len(user_ids)

                # If the number of ratings associated to this item are greater than the number of folds then, this item' ratings can participate in both the training and in the test sets.
                if n >= folds_num:
                    idx = list(range(n))
                    user_ids_folds = random_divide(idx, folds_num)
                    for test_fold in folds_list:
                        # Add users of the current fold as test
                        test_idx = user_ids_folds[test_fold]

                        # Add users of the next fold as validation
                        validation_fold = (test_fold + 1) % folds_num
                        validation_idx = user_ids_folds[validation_fold]

                        # Add the rest as training:
                        train_idx = []
                        for i in folds_list:
                            if i != test_fold and i != validation_fold:
                                train_idx.extend(user_ids_folds[i])

                        train[test_fold][item].extend(user_ids[train_idx].tolist())
                        test[test_fold][item].extend(user_ids[test_idx].tolist())
                        validation[test_fold][item].extend(user_ids[validation_idx].tolist())
                # If the number of ratings associated to this item are less than the number of folds then, this item's ratings can appear in the training set only.
                else:
                    for fold in folds_list:
                        train[fold][item].extend(user_ids.tolist())
                        test[fold][item].extend([])
                        validation[fold][item].extend([])

        # If generating validation set is not required, generate Test and Training sets only:
        else:
            for item in range(self.num_items):
                if item % 5000 == 0:
                    print("doc_{}".format(item))
                user_ids = np.array(items_mat[item])
                n = len(user_ids)

                if n >= folds_num:
                    idx = list(range(n))
                    user_ids_folds = random_divide(idx, folds_num)
                    for test_fold in folds_list:
                        # Add users of the current fold as test
                        test_idx = user_ids_folds[test_fold]

                        # Add the rest as training:
                        train_idx = [id for id in idx if id not in test_idx]
                        train[test_fold][item].extend(user_ids[train_idx].tolist())
                        test[test_fold][item].extend(user_ids[test_idx].tolist())
                else:
                    for fold in folds_list:
                        train[fold][item].extend(user_ids.tolist())
                        test[fold][item].extend([])

        # 2- Generate the user ratings from the splits generated on step 1.
        stats = Stats(self.generate_validation)
        for fold in folds_list:
            items_train = train[fold]
            users_train = self.users_mat_from_items(items_train)

            for u_id, u in enumerate(users_train):
                if len(u) == 0:
                    print("User {} contains 0 training items, split again!".format(u_id))
                    raise Exception("Split_Error!")
            write_ratings(users_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
            write_ratings(items_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

            items_test = test[fold]
            users_test = self.users_mat_from_items(items_test)           
            write_ratings(users_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
            write_ratings(items_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

            if self.generate_validation:
                items_validation = validation[fold]
                users_validation = self.users_mat_from_items(items_validation)
                # Storing the fold validation items for all users
                write_ratings(users_validation, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "validation-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
                write_ratings(items_validation, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "validation-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

            # Calculate statistics:
            if self.generate_validation:
                stats.add_fold_statistics(fold + 1, users_train, users_test, items_train, items_test, users_validation, items_validation)
            else:
                stats.add_fold_statistics(fold + 1, users_train, users_test, items_train, items_test)
            #calculate_split_stats(users_train, users_test, items_train, items_test, fold)

        # Write split statistics:
        stats.save_stats_to_file(os.path.join(self.out_folder, 'stats.txt'))

    def out_of_matrix_split(self, folds_num=5):
        """
        Splits the rating matrix following the out-of-matrix method defined in CTR, the result after invoking this method is:
        two files for each fold (out_of-train-fold_id-users.dat and out_of-train-fold_id-users.dat), both files have the same following format:
        line i has delimiter-separated list of item ids rated by user i
        :param folds_num: the number of folds, default = 5
        :return: None
        """
        # 1- Split items ids in folds:
        items_ids = list(range(self.num_items))
        item_ids_folds = random_divide(items_ids, folds_num)

        # 2- Generate the training and test sets for each fold:
        stats = Stats(self.generate_validation)
        for test_fold in range(folds_num):

            # Get the test, validation and training items:
            items_test_ids = set(item_ids_folds[test_fold])
            items_validation_ids = set()
            if self.generate_validation:
                # Add items of the next fold as validation
                validation_fold = (test_fold + 1) % folds_num
                items_validation_ids = set(item_ids_folds[validation_fold])
            # Add the rest as training:
            items_train_ids = set(items_ids) - items_test_ids - items_validation_ids

            # Generate users ratings for training, test and validation:
            users_train = []
            users_test = []
            users_validation = []

            for user_ratings in self.users_ratings:
                tr_ratings = list(items_train_ids.intersection(user_ratings))
                if len(tr_ratings) == 0:
                    print("some users contains 0 training items, split again again!")
                    raise Exception("Split_Error!")
                tes_ratings = list(items_test_ids.intersection(user_ratings))
                val_ratings = list(items_validation_ids.intersection(user_ratings))

                tr_ratings.sort()
                tes_ratings.sort()
                val_ratings.sort()

                users_train.append(tr_ratings)
                users_test.append(tes_ratings)
                users_validation.append(val_ratings)

            write_ratings(users_train, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "train-fold_{}-users.dat".format(test_fold + 1)), delimiter=self.delimiter)
            write_ratings(users_test, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "test-fold_{}-users.dat".format(test_fold + 1)), delimiter=self.delimiter)
            write_ratings(users_validation, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "validation-fold_{}-users.dat".format(test_fold + 1)), delimiter=self.delimiter)

            items_train = self.items_mat_from_users_ratings(users_train)
            write_ratings(items_train, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "train-fold_{}-items.dat".format(test_fold + 1)), delimiter=self.delimiter)

            items_test = self.items_mat_from_users_ratings(users_test)
            write_ratings(items_test, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "test-fold_{}-items.dat".format(test_fold + 1)), delimiter=self.delimiter)

            items_validation = self.items_mat_from_users_ratings(users_validation)
            write_ratings(items_validation, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "validation-fold_{}-items.dat".format(test_fold + 1)), delimiter=self.delimiter)

            # Saving left out items ids:
            items_test_lst = list(items_test)
            items_test_lst.sort()
            write_ratings(items_test_lst, filename=os.path.join(self.out_folder, "fold-{}".format(test_fold + 1), "heldout-set-fold_{}-items.dat".format(test_fold + 1)), delimiter=self.delimiter, print_line_length=False)

            # Calculate statistics:
            if self.generate_validation:
                stats.add_fold_statistics(test_fold + 1, users_train, users_test, items_train, items_test, users_validation, items_validation)
            else:
                stats.add_fold_statistics(test_fold + 1, users_train, users_test, items_train, items_test)
            # calculate_split_stats(users_train, users_test, items_train, items_test, fold)

        # Write split statistics:
        stats.save_stats_to_file(os.path.join(self.out_folder, 'stats.txt'))

    def user_based_split(self, folds_num=5):
        """
        Splits the rating matrix following the user-based method, the result after invoking this method is:
        two files for each fold (cf-train-fold_id-users.dat and cf-train-fold_id-users.dat), both files have the same format, as following:
        line i has delimiter-separated list of item ids rated by user i        
        :param folds_num: the number of folds, default 5
        :return: None
        """
        train = [[[] for _ in range(self.num_users)] for _ in range(folds_num)]
        test = [[[] for _ in range(self.num_users)] for _ in range(folds_num)]
        for user in range(self.num_users):
            if user % 1000 == 0:
                print("user_{}".format(user))
            items_ids = np.array(self.users_ratings[user])
            n = len(items_ids)
            if n >= folds_num:
                idx = list(range(n))
                item_ids_folds = random_divide(idx, folds_num)
                for fold in range(folds_num):
                    test_idx = item_ids_folds[fold]
                    train_idx = [id for id in idx if id not in test_idx]
                    train[fold][user].extend(items_ids[train_idx].tolist())
                    test[fold][user].extend(items_ids[test_idx].tolist())
            else:
                for fold in range(folds_num):
                    train[fold][user].extend(items_ids.tolist())
                    test[fold][user].extend([])

        stats = Stats(self.generate_validation)
        for fold in range(folds_num):
            users_train = train[fold]
            items_train = self.items_mat_from_users_ratings(users_train)
            for u in users_train:
                if len(u) == 0:
                    print("some users contains 0 training items, split again again!")
                    raise Exception("Split_Error!")
            write_ratings(users_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
            write_ratings(items_train, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "train-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

            users_test = test[fold]
            items_test = self.items_mat_from_users_ratings(users_test)

            # Storing the fold test items for all users
            write_ratings(users_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-users.dat".format(fold + 1)), delimiter=self.delimiter)
            write_ratings(items_test, filename=os.path.join(self.out_folder, "fold-{}".format(fold + 1), "test-fold_{}-items.dat".format(fold + 1)), delimiter=self.delimiter)

            # Calculate statistics:
            #TODO: Calculate Validation sets:
            users_validation = []
            items_validation = []
            if self.generate_validation:
                stats.add_fold_statistics(fold + 1, users_train, users_test, items_train, items_test, users_validation, items_validation)
            else:
                stats.add_fold_statistics(fold + 1, users_train, users_test, items_train, items_test)
            # calculate_split_stats(users_train, users_test, items_train, items_test, fold)

        # Write split statistics:
        stats.save_stats_to_file(os.path.join(self.out_folder, 'stats.txt'))

    def split(self):
        # Calculating and saving the split matrix
        if self.split_method == "user-based":
            self.user_based_split()
            self.create_all_folds_test_split_matrix()
        if self.split_method == "in-matrix-item":
            self.cf_split()
            self.create_all_folds_test_split_matrix()
        if self.split_method == "outof-matrix-item":
            self.out_of_matrix_split()
            self.create_all_folds_test_split_matrix()

        # Write split statistics:
        #if len(self.stats_list) > 1:
        #    with open(os.path.join(self.out_folder, 'stats.txt'), 'w') as f:
        #        f.write("# Users: {}\n# Items: {}\n# Ratings: {}\n".format(self.num_users, self.num_items, sum([len(i) for i in self.users_ratings])))
        #        for s in self.stats_list:
        #            f.write('{}'.format('  '.join(map(str, s)) + '\n'))

if __name__ == '__main__':
    data_folder = "../../data"
    fold_num = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", "-d",
                        help="The folder containing data, the users' ratings file")
    parser.add_argument("--split", "-s", choices=['user-based', 'in-matrix-item', 'outof-matrix-item'],
                        help="The split strategy: uer-based, splits the ratings of each user in train/test; in-matrix-item: CTR in-matrix, outof-matrix-item: CTR out of matrix")
    parser.add_argument("--generate_validation", "-v", action="store_true", default=False,
                        help="if -v is provided, then a validation set will be generated. Default: False, no valiation set is generated")
    parser.add_argument("--fold_num", "-f",type=int,  help="The number of folds to be generated. Default is 5")

    args = parser.parse_args()
    if args.data_directory:
        data_folder = args.data_directory
    if args.fold_num:
        fold_num = args.fold_num
    if args.split:
        splitter = Splitter(data_folder, split_method=args.split, generate_validation=args.generate_validation, folds_num=fold_num)
        splitter.split()