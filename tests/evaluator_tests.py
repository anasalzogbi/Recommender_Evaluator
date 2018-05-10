import unittest
from lib.evaluator import calculate_metrics


class TestEvaluator(unittest.TestCase):
    def runTest(self):
        # TODO: Define the test cases here
        hits = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        recall_breaks = [2, 4, 6, 8, 10, 12]
        mrr_breaks = [2, 5, 10]
        ndcg_breaks = [2, 5, 10]
        results = calculate_metrics(hits, 100, recall_breaks, mrr_breaks, ndcg_breaks)
        row_header = ["Recall@" + str(i) for i in recall_breaks] + ["MRR@" + str(i) for i in mrr_breaks] + [
            "nDCG@" + str(i) for i in ndcg_breaks]
        for (i, j) in (zip(row_header, results)):
            print(i, j)
