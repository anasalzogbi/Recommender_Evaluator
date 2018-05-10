Albert-Ludwigs-Universitaet Freiburg  
Database and Information Systems group  
Georges-koehler-Allee 51, 79110 Freiburg, Germany  
Anas Alzogbi  
email: <alzoghba@informatik.uni-freiburg.de>  



This project provides a python implementation for the split and 
evaluation [system](https://github.com/blei-lab/ctr) used in the work of 
[Collaborative Topic Regression (CTR)](http://www.cs.princeton.edu/~chongw/papers/WangBlei2011.pdf). The purpose of this system is to 
separate the offline evaluation task from model building. It provides 
the following functionality:  
1. Splitting a given users' ratings based on several split methods (Currently supported split methods are: in-matrix-item based, in-matrix-user based, outof-matrix item based and time-aware split) into training and test datasets;
2. Evaluating recommendation predictions over the test dataset based on several evaluation metrics (Currently supported metrics: recall, ndcg, mrr)

### Rquirements:
- python 3.5
- pandas
- numpy


### Train-test Split:
This is implemented in `lib/split.py`, an invocation example:   

    python3  python_code/lib/split.py -d data -s in-matrix-item 
#### Expected arguments:  
`-d [DATA_DIRECTORY]`, the directory must contain items.dat and users.dat files. Files format:
 - `users.dat`: One line for each user, the first value is the number of user's relevant items (n), the next n values are the relevant items ids. The items ids correspond to the line number in the `[items.dat]` file   
 - `items.dat`: One line for each item, the first value is the number of users who like this item (m), the next m values are the relevant users ids. The users ids correspond to the line number in the `[userss.dat]` file 
`-s [SPLIT_STRATEGY]`, The split strategy is one of the following: `user-based`, `in-matrix-item` or `outof-matrix-item`  

#### Expected results:
- New folder: `[DATA_DIRECTORY]`/`[SPLIT_STRATEGY]`_folds
- 5 folder, the 5 folds: `[DATA_DIRECTORY]`/`[SPLIT_STRATEGY]`_folds/fold-`[1-5]`
- Each fold folder contains 4 files:
  - train-fold_`[1-5]`-users.dat
  - test-fold_`[1-5]-`users.dat
  - train-fold_`[1-5]`-items.dat
  - test-fold_`[1-5]`-items.dat
- File contains the splits matrix for test data.  It is a stored `ndarray` of shape: (#num_users, #num_folds, `list-of-test-ids`). The `list-of-test-ids` contains both `user_positive_ids` and `user_fold_unrated_items`:  
`[DATA_DIRECTORY]`/splits.npy
- File contains the split statistics:  
 `[DATA_DIRECTORY]`/`[SPLIT_STRATEGY]`_folds/stats.txt

### Results evaluation:
This is implemented in `lib/evaluator.py`, an invocation example:   

    python3 Recommender_evaluator/lib/evaluator.py -u data/users.dat -s -p data/in-matrix-item_folds -x data/in-matrix-item_folds/CTR_K_200 
#### Expected arguments:  
- `-u [USER_RATINGS_FILE.dat]` users ratings file. Note this is the complete ratings file not split.
- `-s`, a flag indicates to calculate the score matrix as: U.V<sup>T</sup>. `[final-U.dat]` and `[final-V.dat]` files are loaded from `[EXPERIMENT_DIRECTORY]`/fold-`[1-5]`/ Default: False, score matrix is not calculated, assuming the existence of score files: `[EXPERIMENT_DIRECTORY]`/fold-`[1-5]`/score.npy
- `-p [SPLIT_DIRECTORY]` the split direcotry, it should contain the folds' folders (fold-`[1-5]`), each folder contains the test files
- `-x [EXPERIMENT_DIRECTORY]` the experiment directory, it should contain one folder for each fold, each folder contains the models: `[final-U.dat]` and `[final-V.dat]`


#### Expected results:
1. If the argument `-s` is provided, a score file will be saved for each fold under:  
`[EXPERIMENT_DIRECTORY]`/fold-`[1-5]`/score.npy

1. All users results file for each fold, can be used to investigate the results at the user level:  
`[EXPERIMENT_DIRECTORY]`/fold-`[1-5]`/results-users.dat

1. The average results for all folds over all users, for an overview over the results):  
`[EXPERIMENT_DIRECTORY]`/`[EXPERIMENT_NAME]`_eval_results.txt

1. The results matrix, can be used for further statistics over the results are needed. It is a numpy `ndarray` with dimensions: #folds x #users x #metrics:  
`[EXPERIMENT_DIRECTORY]`/results_matrix.npy  

### Directory structure:  
The following directory tree illustrates how the structure of data directory will be:   
|__ `[USER_RATINGS_FILE.dat]`  
|__ `[SPLIT_STRATEGY]`_folds  
......|__ fold`[1-5]`  
......|......|_ train-fold_`[1-5]`-items.dat  
......|......|_ test-fold_`[1-5]`-items.dat  
......|......|_ train-fold_`[1-5]`-users.dat  
......|......|_ test-fold_`[1-5]`-users.dat  
......|__ splits.npy  
......|__ stats.txt  
......|__ `[EXPERIMENT_DIRECTORY]`   
.............|_ `[EXPERIMENT_NAME]`\_eval_results.txt  
.............|_ results-matrix.npy  
.............|_ fold`[1-5]`  
.............|......|_ final-U.dat  
.............|......|_ final-V.dat  
.............|......|_ score.npy  
.............|......|_ results-users.dat   
