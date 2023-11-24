import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC, LinearSVC
import scipy 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.model_selection import cross_val_score
import warnings


def load_sias_scores(path):
    sias_df = pd.read_excel(path+'/Questionnaires.xlsx')
    sias_df = sias_df.drop(sias_df.index[34]) # Deleting subjects not present in pupil_data
    sias_df = sias_df.drop(sias_df.index[24]) 
    sias_df = sias_df.drop(sias_df.index[19])
    sias_df = sias_df.drop(sias_df.index[10])
    sias_df = sias_df.drop(sias_df.index[8])

    sias_score = sias_df.iloc[:, 1:21]
    sias_score = sias_score.apply(pd.to_numeric)
    sias_score['SIAS05'] = [4]*43 - sias_score['SIAS05']
    sias_score['SIAS09'] = [4]*43 - sias_score['SIAS09']
    sias_score['SIAS11'] = [4]*43 - sias_score['SIAS11']
    sias_score = sias_score.sum(axis=1)
    sias_df['score'] = sias_score
    
    thr = int(np.round((np.median(sias_score))))
    sias_df['anxiety'] = sias_score >= thr
    print('\nClass counts:')
    print(np.unique(sias_df['anxiety'].to_numpy(), return_counts=True))
    print('Threshold: ' + str(thr))
    print(' ')
    sias_df = sias_df[['VP', 'score', 'anxiety']]
    return sias_df

def train_sklearn(X, y, model='RF', hyper_search=True):
    if model == 'SVM':
        skmodel = SVC(random_state=1, kernel='rbf')
        distributions = dict(svc__C=scipy.stats.expon(scale=10), svc__gamma=scipy.stats.expon(scale=.1))
    elif model == 'linSVM':
        skmodel = LinearSVC(class_weight='balanced', max_iter=100000)
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


#------------------ MAIN -------------------------

subject_data_path = './data/subject_features.csv'
path_Q = 'data/'

anxiety_data = load_sias_scores(path_Q)

y = anxiety_data['anxiety'].to_numpy()
X = np.array(pd.read_csv(subject_data_path).values.tolist())
X = X[:, 2:]

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Models for the classification
models_classification = ['linSVM', 'SVM', 'RF', 'AdaBoost', 'LogisticRegression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


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

        if(skModel=='RF' and configuration1=='all' and configuration2=='all'):
            feature_importance = model[-1].feature_importances_
            feature_names = ['B′11', 'B′12', 'B′22','B*11', 'B*12', 'B*22','Γ′11','Γ′12','Γ′22','Γ*11','Γ*12','Γ*22', 
                             'B′′11', 'B′′12', 'B′′22','B**11', 'B**12', 'B**22','Γ′′11','Γ′′12','Γ′′22','Γ**11','Γ**12','Γ**22',
                             'shape′','scale′','loc′',
                             'B′′′11', 'B′′′12', 'B′′′22','B***11', 'B***12', 'B***22','Γ′′′11','Γ′′′12','Γ′′′22','Γ***11','Γ***12','Γ***22', 
                             'B′′′′11', 'B′′′′12', 'B′′′′22','B****11', 'B****12', 'B****22','Γ′′′′11','Γ′′′′12','Γ′′′′22','Γ****11','Γ****12','Γ**22',
                             'shape′′','scale′′','loc′′',
                             'shape′′′','scale′′′','loc′′′',
                             'shape′′′′','scale′′′′','loc′′′′',
                             'alpha','gamma','beta','delta',
                             'shape′′′′′','scale′′′′′','loc′′′′′' ]

            plt.figure(figsize=(16, 9))
            plt.title("Feature Importances")
            plt.bar(feature_names, feature_importance, align="center")
            plt.xlabel("Feature")
            plt.ylabel("Feature Importance")
            plt.xticks(rotation=90)
            plt.xticks(fontsize=10)
            plt.show()