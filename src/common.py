#!env python

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from math import sqrt
from matplotlib.patches import Patch
from matplotlib import cm




from math import sqrt
from matplotlib.patches import Patch

import pandas as pd
import sqlite3


from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,  r2_score, make_scorer

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_validate
from scipy import stats

def getSysname():
    return "PieSlicer"

# Gets a default set of markers.  May need to add more markers as needed
def getMarkers():
    return ['+', 'o', 'x', '^', 's', 'v']

# Gets a range of BW colors ranging from dark to light but never including pure black or white
def getBWColors(num_fields, stringify=True):
    if stringify:
        return ["%s" % c for c in getBWColors(num_fields, False)]
    return (1-(np.arange(0,num_fields+1)/float(num_fields+1)))[::-1][:-1]

# Sets Parameters based on "width" of plot.
#   If "width" is set to "markdown" then a reasonably sized PNG file is generated
#   If "width" is set to a float in [0.5, 0.3, 0.25] then a PDF file of this fraction of the page width will be generated
def setRCParams(width=0.5, height=0.5, *args, **kwargs):
    params = {
        "savefig.format"    : 'pdf',
        'text.usetex'       : 'false'
    }
    
    if width == "markdown":
        params['figure.figsize'] = [6.6, 3.0]
        params['font.size'] = (16)
        params['savefig.format'] = 'png'
    elif width == 0.5:
        params['figure.figsize'] = [3.3, 1]
        params['font.size'] = ('%s' % (8*(3/4.)))
    elif width == 0.3:
        params['figure.figsize'] = [2.2, 1.0]
        params['font.size'] = ('%s' % (8*(2/3.)))
    elif width == 0.25:
        params['figure.figsize'] = [2.0, 1.0]
        params['font.size'] = ('%s' % (8*(2./2.)))
    elif width == 0.2:
        params['figure.figsize'] = [2.0, 2.0]
        params['font.size'] = ('%s' % (10))
    elif width == 0.16:
        params['figure.figsize'] = [2.0, 2.0]
        params['font.size'] = ('%s' % (11))
    else:
        params['figure.figsize'] = [3.3, 1.5]
    
    if height == 0.5:
        pass
    elif height == 1.0:
        x, y = tuple(params['figure.figsize'])
        
        params['figure.figsize'] = [x, y*2]
    
        
    matplotlib.rcParams.update(params)

def loadRCParamsFile(path_to_file="matplotlibrc"):
    with open(path_to_file) as fid:
        param_lines = [s.strip() for s in fid.readlines()]

    params = {}
    for line in param_lines:
        if line.strip() == '':
            continue
        if line.strip().startswith('#'):
            continue

        parts = line.split(':')
        key = parts[0].strip()
        value = ':'.join(parts[1:]).strip()
        params[key] = value
    matplotlib.rcParams.update(params)


bin_size = 0.1 # in MB
size_cutoff = 10


## Loading Functions

def load_DB_to_DF(path_to_df, **kwargs):
    print("Loading %s" % path_to_df)
    conn1 = sqlite3.connect(path_to_df)
    df_phone = pd.read_sql_query("SELECT * FROM inference_results;", conn1)
    df_phone = cleanDF(df_phone)
    df_phone = hackDF(df_phone, **kwargs)
    df_phone = addOneHot(df_phone, path_to_df)
    return df_phone

def addOneHot(df, path_to_df):
    for part in os.path.basename(path_to_df).split('.'):
        df[part] = 1
    return df

def cleanDF(df,):
    try:
        df.set_index(pd.DatetimeIndex(df['timeStamp']), inplace=True)
    except KeyError:
        pass
    for col in ["sla_target",
                "orig_size", 
                "sent_size", 
                "image_dims", 
                "time_local_pieslicer",
                "time_local_preprocess",
                'time_local_remote',
                'time_remote_save',
                'time_remote_network',
                'time_remote_transfer',
                'time_remote_pieslicer',
                'time_remote_load',
                'time_remote_general_resize',
                'time_remote_specific_resize',
                'time_remote_convert',
                'time_remote_post_network',
                'time_remote_inference',
                'time_remote_total',
                'time_total',
                'model_accuracy',
                'time_budget',
                'expected_time_local_prep',
                'expected_time_remote_prep',
                'transfer_time_estimate',
                'transfer_time_real',
                'transfer_time_delta',
                'transfer_time_delta_raw',
                'time_local_preprocess_resize',
                'time_local_preprocess_save',
                'time_remote_routing',
                'time_remote_prepieslicer',
                'ping_time',
                'inference_result',
                'time_local_preprocess_check',
                'time_local_preprocess_check_filesize',
                'time_local_preprocess_check_dimensions',
                'preexecution_time_estimate',
                'orig_dims_x',
                'orig_dims_y',
                'jpeg_quality',
               ]:
        try:
            df[col] = df[col].apply(pd.to_numeric, args=('coerce',))
        except KeyError:
            print("Couldn't find col: %s" % col)
    try:
        df["test_image_bool"] = df["test_image_bool"].apply((lambda s: int(s) == 1))
    except KeyError:
        df["test_image_bool"] = False
    del df["_id"]
    df["orig_size"] = df["orig_size"] / 1000000.
    df["sent_size"] = df["sent_size"] / 1000000.
    #df["network_time"] = df["total_time"] - df["total_remotetime"]
    return df

def hackDF(df):
    try:
        df["local_prep"] = df["preprocess_location_real"]=="local"
    except KeyError:
        df["local_prep"] = df["preprocess_location"]=="local"
    
    #df["sla_target_hack"] = (df.index % 100) * 10 + 10
    df["in_sla"] = df["sla_target"] >= df["time_total"]
    df["time_total_network"] = (df["time_local_remote"] - df["time_remote_total"])
    df["time_total_transfer"] = (df["time_local_remote"] - df["time_remote_post_network"])
    df['expected_delta'] = df['expected_time_local_prep'] - df['expected_time_remote_prep']
    df['time_leftover'] = df['sla_target'] - df['time_total']
    df["time_remote_preprocess"] = df["time_remote_post_network"] - df["time_remote_inference"]
    df['effective_bandwidth'] = df['sent_size'] / (df['time_local_remote']/1000.)
    try:
        df['effective_bandwidth_wo_network'] = df['sent_size'] / ((df['time_local_remote'] - df['ping_time'])/1000.)
    except KeyError:
        pass
    try:
        df['time_local_preprocess_check'] = df['time_local_preprocess_check'] / 1000.0 / 1000.0
    except KeyError:
        pass
    
    df["upscaled"] = df["orig_size"] < df["sent_size"]
    
    
    df["preexec_time"] = df["time_local_preprocess"] + (df["time_local_remote"] - df["time_remote_inference"])
    try:
        df["preexec_error"] = df["preexec_time"] - df["preexecution_time_estimate"]
    except KeyError:
        df["preexec_error"] = 0.
        
    df["transfer_time_error"] = df["transfer_time_real"] - df["transfer_time_estimate"]
    
    
    def modelDimFinder(s):
        if "inception" in s:
            return 299
        if "mobilenet" in s:
            return int(s.split('_')[3])
        if "densenet" in s:
            return 224
        if "squeezenet" in s:
            return 224
        if "nasnet" in s:
            if "mobile" in s:
                return 224
            else:
                return 331
        return 0
        
    df["model_dims"] = df["model_name"].apply(modelDimFinder)
    
    df["orig_dims_x"] = df["image_name_orig"].apply( (lambda n: df[df["image_name_orig"]==n]["orig_dims_x"].max()) )
    df["orig_dims_y"] = df["image_name_orig"].apply( (lambda n: df[df["image_name_orig"]==n]["orig_dims_y"].max()) )
    
    try:
        df["pixels"] = df["orig_dims_x"] * df["orig_dims_y"]
    except KeyError:
        pass
    
    df["bin"] = df["orig_size"] - (df["orig_size"] % bin_size)
    
    df = _rejectInput(df)
    
    return df

def _rejectInput(df):
     return df[df["orig_size"] <= size_cutoff]
    
    

def getAlgorithmDF(df_base, algo_name):
    return df_base[df_base["algorithm"]==algo_name].groupby("image_name_orig").mean() #.set_index("image_name_orig")

def calcOptimalDF(df_local, df_remote):
    df_merged = pd.merge(df_local, df_remote, suffixes=("_local", "_remote"), left_index=True, right_index=True)
    df_merged["preexec_time"] = df_merged[["preexec_time_local", "preexec_time_remote"]].min(axis=1)
    df_merged["time_budget"] = df_merged[["time_budget_local", "time_budget_remote"]].max(axis=1)
    df_merged["local_prep"] = df_merged["preexec_time_local"] < df_merged["preexec_time_remote"]
    df_merged["orig_size"] = df_merged["orig_size_local"]
    df_merged["bin"] = df_merged["bin_local"]
    return df_merged

def getBinnedDF(df_base, custom_bin_size=bin_size, col=None, quantiles=None):
    if col is None:
        col = "orig_size"
    df_base["custom_bin"] = df_base[col] - (df_base[col] % custom_bin_size) + custom_bin_size
    if quantiles is None:
        return df_base.groupby("custom_bin").mean(), df_base.groupby("custom_bin").std()
    else:
        return df_base.groupby("custom_bin").mean(), df_base.groupby("custom_bin").quantile(quantiles)


## Modeling functions 
class Modeling():
    @classmethod
    def getTrainTest(cls, X_all, Y_all):
        
        X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.20)
        X_train = X_train.reshape((len(X_train), 1))
        X_test = X_test.reshape((len(X_test), 1))
        
        return X_train, X_test, Y_train, Y_test


    @classmethod
    def getModel(cls, X, y, normalize=True, **kwargs):
        model = LinearRegression(normalize=normalize)
        model.fit(X, y)
        return model
        
    @classmethod
    def getKModels(cls, X, y, n_splits=10, normalize=True, **kwargs):
        model = LinearRegression(normalize=normalize)
        kfold = KFold(n_splits=n_splits, shuffle=True)
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
        return model
    
    @classmethod
    def runKFoldValidation(cls, X_all, Y_all, n_splits=10, normalize=True, **kwargs):
        
        model = LinearRegression(normalize=normalize)
        return cross_val_score(model, X_all, Y_all, cv=n_splits)
    
    @classmethod
    def getKFoldValidationScores(cls, df_to_model, x_col, y_col, n_splits=10):
        if isinstance(y_col, str):
            y_col = [y_col]
        
        if isinstance(x_col, str):
            x_col = [x_col]
        

        df_internal = df_to_model.set_index(x_col)
        
        def getArrays(df_to_use):
            return np.array(df_to_use.index.values.tolist()).reshape(-1,1), df_to_use[y_col].sum(axis=1).values
        
        X_all, Y_all = getArrays(df_internal)
        
        X_all = np.array(X_all)
        Y_all = np.array(Y_all)
        
        model = LinearRegression(normalize=True)
        #scores = cross_validate(model, X, y, cv=n_splits,
        #                 scoring=('neg_mean_squared_error', 'neg_mean_absolute_error'),
        #                 return_train_score=True)
        scores = {}
        scores["RMSE"] = np.sqrt(-1*np.array(cross_val_score(model, X_all, Y_all, cv=n_splits, scoring='neg_mean_squared_error'))).mean()
        scores["MAE"] = np.array(-1*cross_val_score(model, X_all, Y_all, cv=n_splits, scoring='neg_mean_absolute_error')).mean()
        
        
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        scores["MAPE"] = np.array(cross_val_score(model, X_all, Y_all, cv=n_splits, scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=True))).mean()
        
        return scores
        
        

    @classmethod
    def modelDF(cls, df_to_model, x_col, y_col, name_of_df=None, variable_name=None, normalize=True, **kwargs):
        fig, ax = plt.subplots(nrows=1,ncols=1)
        
        if isinstance(y_col, str):
            y_col = [y_col]
        
        if isinstance(x_col, str):
            x_col = [x_col]
        

        df_internal = df_to_model.set_index(x_col)
        
        def getArrays(df_to_use):
            return np.array(df_to_use.index.values.tolist()).reshape(-1,1), df_to_use[y_col].sum(axis=1).values
        
        X_all, Y_all = getArrays(df_internal)
        
        X_all = np.array(X_all)
        Y_all = np.array(Y_all)
        
        k_folds_scores = cls.runKFoldValidation(X_all, Y_all, **kwargs)
        
        model = cls.getModel(X_all, Y_all, normalize=normalize)

        df_internal[y_col].sum(axis=1).plot(ax=ax, linewidth=0, marker='.', **kwargs)
        
        X_range = np.array([min(X_all), max(X_all)])
        ax.plot(X_range, model.predict(X_range.reshape((len(X_range),1))), linewidth=1, marker='', color='k')
        
        ax.set_xlim([0, max(X_all)])
        ax.set_ylim([0, max(Y_all)])
        ax.set_ylabel(cls.fixName(y_col))
        ax.set_xlabel(cls.fixName(x_col))
        
        if variable_name is None:
            eq_str = "f(x) = %.3fx + %.3f" % (model.coef_[0], model.intercept_)
        else:
            eq_str = "%s(x) = %.3fx + %.3f" % (variable_name, model.coef_[0], model.intercept_)
        
        ax.text(0.02*max(X_all), 0.90*max(Y_all), eq_str)
        #ax.text(0.01*max(X_all), 0.77*max(Y_all), "Mean squared error: %.2f" % mean_squared_error(Y_test, model.predict(X_test)))
        ax.text(0.02*max(X_all), 0.75*max(Y_all), "R2 score: %.2f" % r2_score(Y_all, model.predict(X_all)))
        if name_of_df is None:
            plt.savefig( ("images/%s.%s.pdf" % (x_col, y_col,)), bbox_inches='tight')
        else:
            plt.savefig( ("images/%s.%s.%s.pdf" % (name_of_df, x_col, '.'.join(y_col),)), bbox_inches='tight')
        
         
        print("Test Set Average Absolute Error: %0.2f" % ( np.abs(Y_all - model.predict(X_all)).mean()))
        print("Test Set Average Absolute Error %%: %0.2f%%" % ( 100.*np.abs(Y_all - model.predict(X_all)).mean()/Y_all.mean()))
        
        print("Test Set RSME: %0.2f" % ( sqrt(mean_squared_error(Y_all, model.predict(X_all))) ))
        print("Test Set RSME %%: %0.2f%%" % ( 100.*sqrt(mean_squared_error(Y_all, model.predict(X_all)))/Y_all.mean()))
        
        return model, model.score(X_all, Y_all), X_all, Y_all, fig, ax, k_folds_scores
        
    @classmethod
    def modelErrorPercentage(cls, model, X_test, Y_test, bins=0, scale="log", **kwargs):
        #model, _, X_test, Y_test, fig_model, ax_model = cls.modelDF(df_to_model, "orig_size", "time_local_preprocess", "MotoX", variable_name="f")
        error = (model.predict(X_test) - Y_test)
        error_percent = 100.0 * (error / Y_test)
        #error_percent = error

        
        print("Size: %s" % error_percent.size)
        print("Mean: %s %%" % np.mean(error_percent))
        print("StdDev: %s\n %%" % np.std(error_percent))
        
        fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        
        if bins == 0:
            X_to_plot, Y_to_plot = (np.array(list(t)) for t in zip(*sorted(zip(X_test, error_percent))))
        else:
            bin_means, bin_edges, binnumber = stats.binned_statistic(X_test.flatten(), 
                                                                     error_percent.flatten(), 
                                                                     statistic='mean', 
                                                                     bins=bins)
            X_to_plot, Y_to_plot = bin_edges[1:], bin_means
            
        ax.plot( X_to_plot, Y_to_plot )
        ax.set_xscale(scale)
        ax.set_xlim([min(X_test), max(X_test)])
        
        ax.axhline(np.mean(error_percent), color='0.5', linestyle='--')
        
        return fig, ax
        
        
    @classmethod
    def modelError(cls, model, X_test, Y_test, bins=0, scale="log", **kwargs):
        #model, _, X_test, Y_test, fig_model, ax_model = cls.modelDF(df_to_model, "orig_size", "time_local_preprocess", "MotoX", variable_name="f")
        error = (model.predict(X_test) - Y_test)
        
        print("Size: %s" % error.size)
        print("Mean: %s" % np.mean(error))
        print("StdDev: %s\n" % np.std(error))
        
        fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        
        if bins == 0:
            X_to_plot, Y_to_plot = (np.array(list(t)) for t in zip(*sorted(zip(X_test, error))))
        else:
            bin_means, bin_edges, binnumber = stats.binned_statistic(X_test.flatten(), 
                                                                     error.flatten(), 
                                                                     statistic='mean', 
                                                                     bins=bins)
            X_to_plot, Y_to_plot = bin_edges[1:], bin_means
            
        ax.plot( X_to_plot, Y_to_plot )
        ax.set_xscale(scale)
        ax.set_xlim([min(X_test), max(X_test)])
        
        ax.axhline(np.mean(error), color='0.5', linestyle='--')
        
        return fig, ax
        
    @classmethod
    def modelErrorCDF(cls, model, X_test, Y_test, bins=0, scale="log", **kwargs):
        #model, _, X_test, Y_test, fig_model, ax_model = cls.modelDF(df_to_model, "orig_size", "time_local_preprocess", "MotoX", variable_name="f")
        error = (model.predict(X_test) - Y_test)
        error_percent = 100.0 * (error / Y_test)
        #error_percent = error

        
        print("Size: %s" % error_percent.size)
        print("Mean: %s" % np.mean(error_percent))
        print("StdDev: %s\n" % np.std(error_percent))
        
        fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        
        X_to_plot, Y_to_plot = (np.array(list(t)) for t in zip(*sorted(zip(X_test, error_percent))))
        
        counts, bin_edges = np.histogram (Y_to_plot, bins=bins, normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1])#, marker=markers[i], markevery=(num_bins/10))
        max_bin = max(bin_edges)
        min_bin = min(bin_edges)
        
        
        ax.set_ylim([0,1.01])
        ax.set_xlim([min_bin, max_bin])
        
            
        #ax.plot( X_to_plot, Y_to_plot )
        #ax.set_xscale(scale)
        #ax.set_xlim([min(X_test), max(X_test)])
        
        #ax.axhline(np.mean(error_percent), color='0.5', linestyle='--')
        
        return fig, ax


    @classmethod
    def fixName(cls, name):
        if not isinstance(name, str):
            name = ' + '.join(name)
        
        if name == "orig_size":
            name = "Original Size"
        elif name == "preprocess_time_local":
            name = "On-device Preprocessing (ms)"
        elif name == "preprocess_time_remote":
            name = "In-cloud Preprocessing (ms)"
        elif name == "network_time + save_time":
            name = "Network Transfer Time"
        elif name == "Campus":
            name = "Dynamic"
        else:
            name = name.replace('_', ' ')
            name = name.title()
            
        
        if "time" in name.lower():
            name += " (ms)"
        elif "size" in name.lower():
            name += " (MB)"
        elif name == "Mp":
            name = "Image Dimensions (MP)"
        return name

    @classmethod
    def getModelUsagePlot(cls, df_to_plot, **kwargs):
        fig, ax = plt.subplots(nrows=1,ncols=1)
        df_to_plot.groupby("model_name").count()["image_name_orig"].plot(ax=ax, kind='bar', **kwargs)
        return fig, ax

    @classmethod
    def getCDFPlot(cls, dfs_to_plot, cols_to_plot, num_bins=1000, **kwargs):
        
        if isinstance(cols_to_plot, str):
            cols_to_plot = [cols_to_plot]
        
        markers = getMarkers()
        
        max_bin = float('-inf')
        min_bin = float('+inf')
        fig, ax = plt.subplots(nrows=1,ncols=1)
        for i, (df, label) in enumerate(dfs_to_plot[::1]):
            
            data = df[cols_to_plot].sum(axis=1)
                
            counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)
            counts = counts
            
            cdf = np.cumsum(counts)
            cdf = np.insert(cdf, 0, 0.0)
            
            if "complement" in kwargs and kwargs["complement"]:
                ax.plot (bin_edges[:], 1-(cdf/cdf[-1]), label=cls.fixName(label), marker=markers[i], markevery=(num_bins/10))
            else:
                ax.plot (bin_edges[:], (cdf/cdf[-1]), label=cls.fixName(label), marker=markers[i], markevery=(num_bins/10))
                
            max_bin = max([max_bin, max(bin_edges)])
            min_bin = min([min_bin, min(bin_edges)])
        
        ax.legend(loc='best')
        #ax.axvline(sla_target, linestyle='--', color="0.5")
        ax.set_ylim([0,1.01])
        ax.set_xlim([min_bin*0.99, max_bin])
        
        if "complement" in kwargs and kwargs["complement"]:
            ax.set_ylabel("CCDF (%)")
        else:
            ax.set_ylabel("CDF (%)")
            
        ax.set_xlabel(cls.fixName(cols_to_plot))
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='best')
        
        return fig, ax



    
def main():
    pass

if __name__ == '__main__':
    main()
