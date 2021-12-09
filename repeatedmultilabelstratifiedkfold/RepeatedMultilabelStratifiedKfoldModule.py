import numpy as np
import pandas as pd
from iterative_stratification import IterativeStratification # https://github.com/scikit-multilearn/scikit-multilearn/issues/217 

# TODO: Add option to add dataset balancing (class imbalance) by upsampling or downsampling classes (for training only of course)
class RepeatedMultilabelStratifiedKfold:
    def __init__(self, docs, multi_strat_labels, kfolds=4, kfold_reps=10, holdout_ratio=0.1):
        self.holdout_ratio = holdout_ratio
        self.rskf_holdout_test_sets = []
        self.rskf_val_sets = []
        self.kfold_reps = kfold_reps
        self.k = kfolds
        self.train_val_ratio = 1/self.k
        self.multi_strat_labels = multi_strat_labels
        self.docs = docs
        self.setupMultiLabelStratifiedRepeatedKFold()

    def get_multiStratificationLabels(self, docs):
        multi_strat_labels_dict = {key:[] for key in self.multi_strat_labels}
        for doc in docs:
            for key in self.multi_strat_labels:
                if "." in key:
                    keys = key.split(".")
                    val = doc[keys[0]]
                    for k in keys[1:]:
                        val = val[k]
                else:
                    val = doc[key]
                multi_strat_labels_dict[key].append(val)

        return multi_strat_labels_dict

    def get_holdout_indices(self, random_state):

        if self.holdout_ratio == 0:
            return [i for i in range(len(self.docs))], []

        ## multilabel stratified sample
        multi_strat_labels_dict = self.get_multiStratificationLabels(self.docs)
        Y_df = pd.DataFrame(multi_strat_labels_dict)
        Y = np.array(Y_df)
        X = [i for i in range(len(self.docs))]
        X = np.array([X,X]).transpose()

        # X_train, Y_train, X_test, Y_test = iterative_train_test_split(X, Y, self.holdout_ratio)
        n_splits = int(1/self.holdout_ratio)
        k_fold = IterativeStratification(n_splits=n_splits, order=1, random_state=random_state)
        for train, test in k_fold.split(X, Y):
          break

        return list(train), list(test)

    def setupMultiLabelStratifiedRepeatedKFold(self):
        self.rskf_holdout_test_sets = []
        self.rskf_val_sets = []
        for rep in range(self.kfold_reps):
            self.rskf_val_sets.append([])
            # reset seed for each repetition
            # set seed for reproducibility
            random_state = rep*369
            rskf_train_val_sets_indices, rskf_holdout_test_set_indices = self.get_holdout_indices(random_state=random_state)
            self.rskf_holdout_test_sets.append(rskf_holdout_test_set_indices)
            rskf_train_val_docs = [self.docs[i] for i in rskf_train_val_sets_indices]
            multi_strat_labels_dict = self.get_multiStratificationLabels(rskf_train_val_docs)
            Y_df = pd.DataFrame(multi_strat_labels_dict)
            Y = np.array(Y_df)
            X = np.array([rskf_train_val_sets_indices,rskf_train_val_sets_indices]).transpose()
            n_splits = int(1/self.train_val_ratio)
            k_fold = IterativeStratification(n_splits=n_splits, order=1, random_state=random_state)
            for train, test in k_fold.split(X, Y):
              self.rskf_val_sets[rep].append(list(X[test, 0]))

        # check no reps are redundant....
        duplicate_exists, duplicate_reps = self.checkIfDuplicates(self.rskf_holdout_test_sets)
        return

    def checkIfDuplicates(self, listOfElems):
      ''' Check if any hold out test sets have duplicates'''
      duplicate_reps = []    
      setOfElems = set()
      for rep, elem in enumerate(listOfElems):
          elem.sort()
          elem = [str(i) for i in elem]
          elem = "-".join(elem)
          if elem in setOfElems:
              duplicate_reps.append(rep)
          else:
              setOfElems.add(elem)         
      
      if duplicate_reps:
        print("Duplicate holdout test sets found. Possibility of a repetition to be redundant. Consider reducing the number of repetition")
        print("10 repetitions for a 4-fold CV provides 40(10x4) estimates which is usually enough for most experiments")
        print("duplicate_reps", duplicate_reps)
        # print("The following was tested and worked: single-class stratification")
        '''
        The following was tested and worked: 
        6-label stratification, data size 500, reps = 100, kfolds = 4  
        '''
        return True, duplicate_reps
      
      return False, duplicate_reps

    def getDataSplits(self, rep, fold, verbose=False):
        train_set, val_set, holdout_test_set = [], [], []
        for i, doc in enumerate(self.docs):

            if i in self.rskf_holdout_test_sets[rep] and i in self.rskf_val_sets[rep][fold]:
              print("OH NOOOOOOOO!!!!!!!!!")

            if i in self.rskf_holdout_test_sets[rep]:
                holdout_test_set.append(doc)
            elif i in self.rskf_val_sets[rep][fold]:
                val_set.append(doc)
            else:
                train_set.append(doc)

        if verbose:
            cv_set_size = len(train_set) + len(val_set)
            print("Repetition", rep, "Fold", fold)
            print('Training:  ', len(train_set), len(train_set) / cv_set_size)
            print('Validation: ', len(val_set), len(val_set) / cv_set_size)
            print('Test:      ', len(holdout_test_set), len(holdout_test_set) / len(self.docs))

        return train_set, val_set, holdout_test_set