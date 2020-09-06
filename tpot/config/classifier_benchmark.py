# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

# Check the TPOT documentation for information on the structure of config dicts
import sklearn

tpot_config = {

    # Classifiers
    'sklearn.ensemble.AdaBoostClassifier': {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'learning_rate': [0.1, 0.25, 0.5, 1, 1.5, 2],
        'algorithm': ['SAMME.R', 'SAMME'],
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),  # TODO value not aligned with search space
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier': {
        'loss': ['auto'],
        'learning_rate': [1e-2, 1e-1, 0.5, 1.],
        'min_samples_leaf': [1, 5, 10, 20, 100, 200],
        'max_depth': [None],
        'max_leaf_nodes': [3, 10, 100, 1000, 2047],
        'max_bins': [255],
        'l2_regularization': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1],
        'tol': [1e-7],
        'scoring': ['loss'],
        'n_iter_no_change': [1, 5, 10, 15, 20],
        'validation_fraction': [0.01, 0.1, 0.2, 0.3, 0.4]
    },

    'sklearn.svm.SVC': {
        'C': [0.03125, 1, 10, 100, 1000, 32768],
        'kernel': ["rbf", "poly", "sigmoid"],
        'degree': range(2, 6),
        'gamma': [3.0517578125e-05, 1e-3, 1e-1, 1, 3, 8],
        'coef0': [-1, -0.5, 0, 0.5, 1],
        'shrinking': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'max_iter': [-1]
    },

    'sklearn.discriminant_analysis.LinearDiscriminantAnalysis': {
        'solver': ["svd", "lsqr"],
        'shrinkage': [0, 0.25, 0.5, 0.75, 1],
        'n_components': [1, 10, 50, 125, 250],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [512],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'max_depth': [None],
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'min_weight_fraction_leaf': [0.],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0.],
        'bootstrap': [True, False]
    },

    'sklearn.linear_model.SGDClassifier': {
        'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
        'penalty': ["l1", "l2", "elasticnet"],
        'alpha': [1e-7, 1e-5, 1e-3, 1e-2, 1e-1],
        'l1_ratio': [1e-9, 1e-7, 1e-5, 1e-3, 1],
        'fit_intercept': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'epsilon': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'learning_rate': ['optimal', 'invscaling', 'constant'],
        'eta0': [1e-7, 1e-5, 1e-3, 1e-1],
        'power_t': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        'average': [True, False]
    },

    # Data Preprocessing
    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
    },

    'sklearn.preprocessing.QuantileTransformer': {
        'n_quantiles': [10, 100, 1000, 2000],
        'output_distribution': ['uniform', 'normal']
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    # Feature Preprocessing
    'sklearn.neural_network.BernoulliRBM': {
        'n_components': [1, 16, 32, 64, 128, 256, 512],
        'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    },

    'sklearn.preprocessing.Binarizer': {
        'threshold': [0]
    },

    'sklearn.decomposition.FactorAnalysis': {
        'n_components': [1, 5, 10, 50, 100, 200, 250],
        'max_iter': [10, 50, 100, 1000, 2000],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'svd_method': ["lapack", "randomized"],
        'iterated_power': [1, 3, 5, 8, 10]
    },

    'sklearn.decomposition.FastICA': {
        'n_components': [10, 100, 1000, 2000],
        'algorithm': ['parallel', 'deflation'],
        'whiten': ['False', 'True'],
        'fun': ['logcosh', 'exp', 'cube']
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'n_clusters': [2, 10, 50, 100, 200, 400],
        "affinity": ["euclidean", "manhattan", "cosine"],
        'linkage': ['ward', 'complete', 'average'],
        "pooling_func": [np.mean, np.median, np.max]
    },

    'sklearn.feature_selection.GenericUnivariateSelect': {
        'mode': ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'],
        'param': [1e-5, 1e-2, 0.1, 0.25, 0.75],
        'score_func': [sklearn.feature_selection.chi2, sklearn.feature_selection.f_classif,
                       sklearn.feature_selection.f_regression]
    },

    'sklearn.preprocessing.KBinsDiscretizer': {
        'n_bins': [2, 5, 10, 25, 50, 100],
        'encode': ["onehot", "onehot-dense", "ordinal"],
        'strategy': ["uniform", "quantile", "kmeans"]
    },

    'sklearn.decomposition.KernelPCA': {
        "n_components": [10, 100, 1000, 2000],
        'kernel': ['poly', 'rbf', 'sigmoid', 'cosine'],
        'gamma': [3.0517578125e-05, 1e-3, 1e-1, 1, 3, 8],
        'degree': range(2, 6),
        'coef0': [-1, -0.5, 0, 0.5, 1]
    },

    'sklearn.impute.MissingIndicator': {
        'features': ["missing-only", "all"]
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },

    'sklearn.decomposition.PCA': {
        'keep_variance': [0.5, 0.65, 0.8, 0.9, 0.9999],
        'whiten': [True, False]
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2, 3],
        'include_bias': [True, False],
        'interaction_only': [True, False]
    },

    'sklearn.ensemble.RandomTreesEmbedding': {
        'n_estimators': [10, 25, 50, 75, 100],
        'max_depth': [2, 3, 5, 8, 10],
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'min_weight_fraction_leaf': [1.],
        'max_leaf_nodes': [None],
        'bootstrap': [True, False]
    },

    'sklearn.feature_selection.SelectKBest': {
        'k': [1, 2, 4, 8, 16, 32, 64, 128],
        'score_func': [sklearn.feature_selection.chi2, sklearn.feature_selection.f_classif,
                       sklearn.feature_selection.mutual_info_classif]
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': [sklearn.feature_selection.chi2, sklearn.feature_selection.f_classif,
                       sklearn.feature_selection.mutual_info_classif]
    },

    'sklearn.decomposition.TruncatedSVD': {
        'n_components': [10, 32, 64, 128, 256]
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    }
}
