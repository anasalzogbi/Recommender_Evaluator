"""
Author: Anas Alzogbi
Description: this module provides the functionality of:
 - Calculating statistics for splits
 - Saving the calculated statistics in a well-formated way into a file.
Date: December 18th, 2017
"""

import numpy as np


class Stats(object):
    def __init__(self, validation=False):
        self.stats_list = []
        self.text_list=[]
        if validation:
            stats_header = ['{:4}'.format('Fold'), '{:27}'.format('#Usrs(Tot,R,S,V)'), '{:31}'.format('#Itms(Tot,R,S,V)'), '{:31}'.format('#Rtng(Tot,R,S,V)'), \
                            '{:23}'.format('PRU(min/max/avg/std)'), '{:22}'.format('PSU(min/max/avg/std)'), '{:22}'.format('PVU(min/max/avg/std)'), '{:21}'.format('PRI(min/max/avg/std)'),
                            '{:20}'.format('PSI(min/max/avg/std)'), '{:20}'.format('PVI(min/max/avg/std)')]
        else:
            stats_header = ['{:4}'.format('Fold'), '{:20}'.format('#Usrs(Tot,R,S)'), '{:23}'.format('#Itms(Tot,R,S)'), '{:23}'.format('#Rtng(Tot,R,S)'), \
                            '{:23}'.format('PRU(min/max/avg/std)'), '{:22}'.format('PSU(min/max/avg/std)'), '{:20}'.format('PRI(min/max/avg/std)'), '{:20}'.format('PSI(min/max/avg/std)')]
        self.stats_list.append(stats_header)
        self.validation = validation

    def add_fold_statistics(self, fold_num, train_users, test_users, train_items, test_items, validation_users=None, validation_items=None):
        """
        Calculates and adds new statistics for the given fold information to the statistic list
        :param fold_num: The number of the fold associated with the statistics, 1-based
        :param train_users: the users' training data: list of lists
        :param test_users:  the users' test data: list of lists
        :param train_items:  the items' training data: list of lists
        :param test_items:  the items' test data: list of lists
        :param validation_users:  the users' validation data: list of lists. Default: None
        :param validation_items:  the users' validation data: list of lists. Default: None
        """
        tru = [len(i) for i in train_users]
        tsu = [len(i) for i in test_users]
        tri = [len(i) for i in train_items]
        tsi = [len(i) for i in test_items]
        if self.validation and validation_users is not None and validation_items is not None:
            vu = [len(i) for i in validation_users]
            vi = [len(i) for i in validation_items]
            num_users = sum([1 if i > 0 or j > 0 or k > 0 else 0 for (i, j,k) in zip(tru, tsu, vu)])
            num_items = sum([1 if i > 0 or j > 0 or k > 0 else 0 for (i, j, k) in zip(tri, tsi, vi)])
            users_validation = sum([1 if i > 0 else 0 for i in vu])
            items_validation = sum([1 if i > 0 else 0 for i in vi])
        else:
            num_users = sum([1 if i > 0 or j > 0 else 0 for (i, j) in zip(tru, tsu)])
            num_items = sum([1 if i > 0 or j > 0 else 0 for (i, j) in zip(tri, tsi)])
        users_train = sum([1 if i > 0 else 0 for i in tru])
        users_test = sum([1 if i > 0 else 0 for i in tsu])

        items_train = sum([1 if i > 0 else 0 for i in tri])
        items_test = sum([1 if i > 0 else 0 for i in tsi])
        if self.validation and validation_users is not None and validation_items is not None:
            self.stats_list.append(['{:4}'.format(fold_num), '{:5d} / {:5d} / {:4d} / {:4d}'.format(num_users, users_train, users_test, users_validation),\
                                    '{:6d} / {:6d} / {:5d} / {:5d}'.format(num_items, items_train, items_test, items_validation),
                                    '{:6d} / {:6d} / {:5d} / {:5d}'.format(sum(tru) + sum(tsu)+sum(vu), sum(tru), sum(tsu), sum(vu)),
                                    '{:1d} / {:4d} / {:4.1f} / {:5.1f}'.format(np.min(tru), np.max(tru), np.mean(tru), np.std(tru)),
                                    '{:1d} / {:4d} / {:4.1f} / {:4.1f}'.format(np.min(tsu), np.max(tsu), np.mean(tsu), np.std(tsu)),
                                    '{:1d} / {:4d} / {:4.1f} / {:4.1f}'.format(np.min(vu), np.max(vu), np.mean(vu), np.std(vu)),
                                    '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(tri), np.max(tri), np.mean(tri), np.std(tri)),
                                    '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(tsi), np.max(tsi), np.mean(tsi), np.std(tsi)),
                                    '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(vi), np.max(vi), np.mean(vi), np.std(vi))])
        else:
            self.stats_list.append(['{:4}'.format(fold_num), '{:5d} / {:5d} / {:4d}'.format(num_users, users_train, users_test),
                                    '{:6d} / {:6d} / {:5d}'.format(num_items, items_train, items_test),
                                    '{:6d} / {:6d} / {:5d}'.format(sum(tru) + sum(tsu), sum(tru), sum(tsu)),
                                    '{:1d} / {:4d} / {:4.1f} / {:5.1f}'.format(np.min(tru), np.max(tru), np.mean(tru), np.std(tru)),
                                    '{:1d} / {:4d} / {:4.1f} / {:4.1f}'.format(np.min(tsu), np.max(tsu), np.mean(tsu), np.std(tsu)),
                                    '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(tri), np.max(tri), np.mean(tri), np.std(tri)),
                                    '{:1d} / {:3d} / {:4.1f} / {:3.1f}'.format(np.min(tsi), np.max(tsi), np.mean(tsi), np.std(tsi))])

    def add_to_text_list(self, text):
        self.text_list.append(text)
    def save_stats_to_file(self, output_file):
        """
        Write split statistics to file. Invoke this after adding all statistics.
        :param output_file: The output file.
        """
        if len(self.stats_list) > 1:
            with open(output_file, 'w') as f:
                for s in self.stats_list:
                    f.write('{}'.format(' | '.join(map(str, s)) + '\n'))
                if len(self.text_list)>0:
                    f.write('\n------------------------------\n')
                    for s in self.text_list:
                        f.write(s+'\n')
                f.write('\n------------------------------\n')
                f.write('Legend:\n')
                f.write('#Usrs: Number of Users\n')
                f.write('#Itms: Number of Items\n')
                f.write('#Rtng: Number of Ratings\n')
                f.write('(Tot,R,S): Total, Training, Test\n')
                f.write('PRU: Positives in Training per User\n')
                f.write('PSU: Positives in Test per User\n')
                f.write('PRI: Positives in Training per Item\n')
                f.write('PSI: Positives in Test per Item\n')
                if self.validation:
                    f.write('PVU: Positives in Validation per User\n')
                    f.write('PVI: Positives in Validation per Item\n')

