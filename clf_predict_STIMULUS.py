from os.path import join
import scipy
import os
import numpy as np
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
import numpy_indexed as npi
from sklearn.svm import LinearSVR, LinearSVC, OneClassSVM, SVR, SVC
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import Nystroem
from scipy.stats import uniform
import pandas as pd
import re
from sklearn.model_selection import cross_val_score


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)



def train_sklearn(X, y, model='RF', hyper_search=True):
    if model == 'SVM':
        skmodel = SVC(random_state=1, kernel='rbf')
        distributions = dict(svc__C=scipy.stats.expon(scale=10), svc__gamma=scipy.stats.expon(scale=.1))
    elif model == 'linSVM':
        skmodel = LinearSVC(class_weight='balanced', max_iter=10000000)
        distributions = dict(linearsvc__C=scipy.stats.expon(scale=10))
    elif model == 'RF':
        skmodel = RandomForestClassifier(max_depth=2, class_weight='balanced_subsample', random_state=0)
        distributions = {'randomforestclassifier__bootstrap': [True, False],
               'randomforestclassifier__max_depth': [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, None],
               #'randomforestclassifier__max_features': ['auto', 'sqrt'],
               'randomforestclassifier__min_samples_leaf': [2, 4, 6],
               'randomforestclassifier__min_samples_split': [2, 5, 8],
               'randomforestclassifier__n_estimators': [50, 100, 150]}
    elif model == 'AdaBoost':
        skmodel = AdaBoostClassifier()
        distributions = {
            'adaboostclassifier__n_estimators': [50, 100, 150],
            'adaboostclassifier__learning_rate': [0.01, 0.1, 0.2]}
    elif model == 'LogisticRegression':
        skmodel = LogisticRegression(solver='liblinear', class_weight='balanced')
        distributions = {
            'logisticregression__C': scipy.stats.expon(scale=1),
            'logisticregression__penalty': ['l1', 'l2']}

    pipe = make_pipeline(StandardScaler(),
                         skmodel)

    gs = RandomizedSearchCV(pipe,
                            distributions,
                            scoring='f1',
                            n_iter=1000,
                            n_jobs=-1,
                            cv=3)
    if hyper_search:
        gs = gs.fit(X, y)
        print('Best parameters: ', gs.best_params_)
        score = gs.score(X, y)
        print('\tF1: ' + str(score))
        clf = gs.best_estimator_
        return clf, gs
    else:
        pipe_svc = pipe_svc.fit(X, y)
        score = pipe_svc.score(X, y)
        print('\tAccuracy: ' + str(score))
        return pipe_svc, pipe_svc

def load_stimulus(behavior_path):
    paths = os.listdir(behavior_path)
    map_shock_exp = {}

    for path in paths:
        look_df = pd.read_csv(behavior_path + path, sep='\t')
        for x in look_df.iterrows():
            map_shock_exp[(x[1]['subject'], x[1]['trial'])] = x[1]['shock']

    return map_shock_exp


def get_features(X, configuration1, configuration2):
    if configuration1 == 'all':
        if configuration2 == 'all':
            return X
        elif configuration2 == 'ou':
            return np.concatenate((X[:, 0:24], X[:, 27:51], X[:, -3:]), axis=1)
        elif configuration2 == 'classic':
            return np.concatenate((X[:, 24:27], X[:, 51:]), axis=1)
        
    elif configuration1 == 'fix':
        if configuration2 == 'all':
            return X[:, 0:27]
        elif configuration2 == 'ou':
            return X[:, 0:24]
        elif configuration2 == 'classic':
            return X[:, 24:27]
    
    elif configuration1 == 'sac':
        if configuration2 == 'all':
            return X[:, 27: -3]
        elif configuration2 == 'ou':
            return X[:, 27:51]
        elif configuration2 == 'classic':
            return X[:, 51:-3]
        
    elif configuration1 == 'pupil':
        return X[:, -3:]

def load_id_to_keep(): # pupil data contain only 43 subjects
    list_pupil = os.listdir('./data/pupil_data/')
    id_tokeep = [ int(s.split('_')[-1].split('.')[0].split('l')[1]) for s in list_pupil ]
    return id_tokeep

# MAIN ---------------------------------------------------------------------

trial_data_path = './data/trial_features.csv'
map_shock_exp = load_stimulus('./data/behavior/')

models_classification = ['SVM', 'RF', 'AdaBoost', 'LogisticRegression']

# Creating train and test sets

id_tokeep = load_id_to_keep()
train_ids, test_ids = train_test_split(id_tokeep, test_size=0.2)

X = np.array(pd.read_csv(trial_data_path).values.tolist())
sbj_trial = X[:, 1:3]
y = np.array([map_shock_exp[(t[0], t[1])] for t in sbj_trial])

print('\nClass counts:')
print(np.unique(y, return_counts=True))

condition_train = np.isin( [int(x) for x in X[:, 1] ], train_ids )
X_train = X[condition_train]
y_train = y[condition_train]

condition_test = np.isin( [int(x) for x in X[:, 1] ], test_ids )
X_test = X[condition_test]
y_test = y[condition_test]


X = X[:, 3:]
X_train = X_train[:, 3:]
X_test = X_test[:, 3:]

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train, y_train = undersampler.fit_resample(X_train, y_train)


for models, configuration1, configuration2 in  [ (models_classification, 'all', 'all'),
                                                 (models_classification, 'all', 'ou'),
                                                 (models_classification, 'all', 'classic'),
                                                 (models_classification, 'fix', 'all'),
                                                 (models_classification, 'fix', 'ou'),
                                                 (models_classification, 'fix', 'classic'),
                                                 (models_classification, 'sac', 'all'),
                                                 (models_classification, 'sac', 'ou'),
                                                 (models_classification, 'sac', 'classic'), 
                                                 (models_classification, 'pupil', '_') ]:


    print('Selecting the current features')

    X_selected = get_features(X, configuration1, configuration2)
    X_train_selected = get_features(X_train, configuration1, configuration2)
    X_test_selected = get_features(X_test, configuration1, configuration2)


    print('Training and testing models with ', configuration1+' ,  '+configuration2 )


    for skModel in models:

        print('\nTraining ONE model... (' + skModel + ')')
        model, gs = train_sklearn(X_train_selected, y_train.astype(int), model=skModel, hyper_search=True)

        print('\nTesting...')
        y_pred_train = model.predict(X_train_selected)
        y_pred_test = model.predict(X_test_selected)

        f1score = f1_score(y_true=y_test.astype(int), y_pred=y_pred_test.ravel())
        acc_score = accuracy_score(y_true=y_test.astype(int), y_pred=y_pred_test.ravel())

        print('\nTest Accuracy Score: ' + str(acc_score))
        print('\nTest F1 Score: ' + str(f1score))
        print(' ')

        print('\nTrain labels: ')
        print(y_train.astype(int))
        print(y_pred_train)

        print('\nTest labels: ')
        print(y_test.astype(int))
        print(y_pred_test)


        print('\nCross Validation:')
        cvs = cross_val_score(gs, X_selected, y.astype(int), cv=5, scoring='accuracy')
        print(cvs)
        print('Average CV score: ' + str(np.mean(cvs)))
        print(' ')
