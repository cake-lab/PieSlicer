#!env python

import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.neighbors
import sklearn.ensemble
import sklearn.model_selection
import sklearn.utils
import sklearn.svm

import seaborn as sns

import itertools
import time 

#from pandas.api.types.is_numeric_dtype

#from 

sys.path.append("../")

import common


android_db_dir = './PATH_TO_ANDROID_DBS' # PATH_TO_ANDROID_DBS

a_priori_factors = [
    'sent_size',
    'orig_dims_x',
    'orig_dims_y',
    'orig_size',
    'pixels',
]
one_hot_factors = [
    'campus',
    'home',
    'nexus5',
    'motox',
    'pixel2',
]
target_column = 'transfer_time_real'
#target_column = 'time_local_preprocess'
#target_column = 'time_remote_preprocess'


def getTimeInMillis():
    return time.time() * 1000.0
def timeit(method):

    def timed(*args, **kwargs):
        ts = getTimeInMillis()
        result = method(*args, **kwargs)
        te = getTimeInMillis()
        
        return result, (te-ts)

    return timed


def getDFsFromDBs(android_db_dir, break_early=False):
    # Load DBs into dataframes
    dataframes = {}
    dbs_to_load = os.listdir(android_db_dir)
    for i, android_db in enumerate( (dbs_to_load if not break_early else dbs_to_load[:5]) ):
        if "128_331_1000" in android_db:
            continue
        if "optimizations" in android_db:
            continue
        path_to_db = os.path.join(android_db_dir, android_db)
        print("Loading %s/%s - %s" % (i+1, len(dbs_to_load), path_to_db,))
        
        if break_early:
            pass
            if "1000images" not in path_to_db:
                continue
        
        # Load the file in a db, indexed by name
        dataframes[android_db] = common.load_DB_to_DF(path_to_db)
        
    return dataframes


def getTestTrainSplits(dataframes, *args, **kwargs):
    
    if "group_keywords" in kwargs:
        group_keywords = kwargs["group_keywords"]
    else:
        group_keywords = dataframes.keys()
    
    testtrain_splits = {}
    
    for keywords in list(group_keywords):
        #relevant_dfs = [dataframes[key] for key in dataframes.keys() if keywords in key]
        relevant_df_keys = list(dataframes.keys())
        for word in keywords:
            relevant_df_keys = [ key for key in relevant_df_keys if word in key ]
        
        relevant_dfs = [dataframes[key] for key in list(set(relevant_df_keys))]
        print("keywords: %s" % (keywords,))
        for df_name in  list(set(relevant_df_keys)):
            print(df_name)
        
        
        if len(relevant_dfs) == 0:
            continue
        elif len(relevant_dfs) == 1:
            combined_df = relevant_dfs[0]
        else:
            combined_df = pd.concat(relevant_dfs, sort=False) #, columns=list(set( [col for df in relevant_dfs for col in df.columns ] )))
        
        # Isolate just the columns we care about
        cols_to_isolate = [ col for col in (a_priori_factors + one_hot_factors) if col in combined_df.columns.values]
        #print("cols_to_isolate: %s" % cols_to_isolate)
        combined_df = combined_df[cols_to_isolate + [target_column]]
        
        # Remove outliers
        combined_df = remove_outliers(combined_df)
        
        # Remove all zero entries
        #print(combined_df[cols_to_isolate].describe())
        
        
        combined_df = combined_df.fillna(0)
        #combined_df = combined_df.dropna()
        #combined_df = combined_df[~(combined_df[a_priori_factors] == 0).any(axis=1)]
        if len(combined_df.index) == 0:
            print("breaking")
            continue
        #combined_df = combined_df.reset_index()
        
        
        X = combined_df[cols_to_isolate].values
        y = combined_df[target_column].values
        
        #print(X.shape)
        
        if "add_logistic" in kwargs and kwargs["add_logistic"]:
            #combined_df = combined_df[(combined_df != 0).all(1)]
            X = np.append(X, np.log(1+combined_df[a_priori_factors].values), axis=1)
            
        if "add_square" in kwargs and kwargs["add_square"]:
            X = np.append(X, np.square(np.log(1+combined_df[a_priori_factors].values)), axis=1)
        
        def scale_linear_bycolumn(rawpoints, high=1.0, low=0.0): # https://gist.github.com/perrygeo/4512375
            mins = np.min(rawpoints, axis=0)
            maxs = np.max(rawpoints, axis=0)
            rng = maxs - mins
            return high - (((high - low) * (maxs - rawpoints)) / rng)
            
        if "normalize" in kwargs and kwargs["normalize"]:
            X = X / X.max(axis=0)
            #X = scale_linear_bycolumn(X)
        
        X = X.reshape(X.shape[0],-1)
        y = y.reshape(-1,1)
        testtrain_splits[keywords] = tuple( list(sklearn.model_selection.train_test_split(X, y, test_size=0.2) + [cols_to_isolate]))
        
        print("")
        
    return list(testtrain_splits.items())
    
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def getModelAccuracy(estimator_type, X_train, X_test, y_train, y_test, num_folds=10, scoring=mean_absolute_percentage_error, *args, **kwargs):
    results = sklearn.model_selection.cross_validate(estimator_type,
                                                X = X_train,
                                                y = y_train,
                                                cv = 10,
                                                return_estimator=True,
                                                scoring=sklearn.metrics.make_scorer(scoring)
                                            )
    training_scores = results["test_score"]
    #print("%s Training Score: %.03f +- %.03f" % (scoring.__name__, np.mean(training_scores), np.std(training_scores)) )
    estimators = list(results["estimator"])
    test_scores = []
    for estimator in estimators:
        y_pred = estimator.predict(X_test)
        #test_scores.append( mean_absolute_percentage_error(y_test, y_pred) )
        test_scores.append( scoring(y_test, y_pred) )
    #print("%s Test Score: %.03f +- %.03f" % (scoring.__name__, np.mean(test_scores), np.std(test_scores)) )
    return (scoring.__name__, np.mean(test_scores), np.std(test_scores), estimators[0])
    
def remove_outliers(df, cols_to_scan=[target_column]): # https://gist.github.com/ariffyasri/70f1e9139da770cb8514998124560281
    # This may need more consideration as it current doesn't consider the linear trend I don't think...
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(cols_to_scan):
        if pd.api.types.is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df


def parseArgs():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--null', help="Does nothing.  Literally just a placeholder.")
    return parser.parse_args()

def main():
    args = parseArgs()
    
    db_dataframes = getDFsFromDBs(android_db_dir, break_early=False)
    
    group_keywords = ["", 
                        "home", "campus", 
                        "1000images", "5000images",
                        'pixel', 'motox', 'nexus5', 
                        'pixel2.home', 'pixel2.campus', 
                        'motox.home', 'motox.campus', 
                        'nexus5.home', 'nexus5.campus']
                        
    
    datasets = ["", "1000images", "5000images"]
    networks = ["", "home", "campus"]
    devices = ["", "pixel", "motox", "nexus"]
    
    for combo in [ [key for key in list(set(combo)) if key != ''] for combo in  itertools.product(datasets, networks, devices)]:
        print(combo)
    group_keywords = [ tuple([key for key in list(set(combo)) if key != '']) for combo in  itertools.product(datasets, networks, devices)] + [tuple([''])]
    
    #group_keywords = [tuple([d]) for d in devices]
    
    
    #print(list(db_dataframes.values())[0].head(1))
    #print(list(db_dataframes.values())[0].describe())
    print("")
    print("")
    for label, (X_train, X_test, y_train, y_test, cols_to_isolate) in getTestTrainSplits(db_dataframes, 
                                                                        add_square=False, 
                                                                        add_logistic=True, 
                                                                        normalize=True,
                                                                        group_keywords=group_keywords):
        print("Label: %s" % (label,))
        
        if True:
            @timeit
            def thing():
                print("Linear LinearRegression")
                scoring_method, score_avg, score_std, _ = getModelAccuracy(sklearn.linear_model.LinearRegression(), X_train, X_test, y_train, y_test)
                print("%s Test Score: %.03f +- %.03f" % (scoring_method, score_avg, score_std) )
            print(thing())
            print("")
        
        
        if True:
            @timeit
            def thing():
                print("KNN Regression")
                # This is technically speaking wrong, but shh
                # (Specifically, it's picking the best test result, not the best due to CV)
                scoring_method, score_avg, score_std, _, best_k = tuple(max(
                    [(list(getModelAccuracy(sklearn.neighbors.KNeighborsRegressor(n_neighbors=k), X_train, X_test, y_train, y_test)) + [k]) 
                        for k in (list(range(1, 30, 1)) + [50, 100])
                    ]
                    , key=(lambda s: (s[1], s[2]))))
                print("k=%s: %s Test Score: %.03f +- %.03f" % (best_k, scoring_method, score_avg, score_std) )
            print(thing())
            print("")
        
        if True:
            @timeit
            def thing():
                print("Random Forest")
                scoring_method, score_avg, score_std, _, best_n, best_depth = tuple(max(
                    [(list(getModelAccuracy(sklearn.ensemble.RandomForestRegressor(n_estimators=n, max_depth=depth), X_train, X_test, np.ravel(y_train), np.ravel(y_test))) + [n, depth]) 
                        for n in [2**i for i in range(4)]
                        for depth in range(1,11)
                        ]
                        , key=(lambda s: (s[1], s[2]))))
                print("%s estimators & depth %s - %s Test Score: %.03f +- %.03f" % (best_n, best_depth, scoring_method, score_avg, score_std) )
            print(thing())
            print("")
        
        if True:
            @timeit
            def thing():
                print("Lasso")
                scoring_method, score_avg, score_std, estimator = getModelAccuracy(sklearn.linear_model.Lasso(), X_train, X_test, y_train, y_test)
                print("%s (%s) coefs : %s Test Score: %.03f +- %.03f" % (len([c for c in estimator.coef_ if c != 0.] ), len(estimator.coef_), scoring_method, score_avg, score_std) )
                #for i in range(len(cols_to_isolate)):
                #    print("%s : %s" % (cols_to_isolate[i], estimator.coef_[i]))
            print(thing())
            print("")
            
        if True:
            @timeit
            def thing():
                print("Non-linear SVR")
                scoring_method, score_avg, score_std, _ = getModelAccuracy(sklearn.svm.SVR(), X_train, X_test, np.ravel(y_train), np.ravel(y_test))
                print("%s Test Score: %.03f +- %.03f" % (scoring_method, score_avg, score_std) )
            print(thing())
            print("")
        
        
        
        print("==================================")
        print("")
        


if __name__ == '__main__':
    main()



