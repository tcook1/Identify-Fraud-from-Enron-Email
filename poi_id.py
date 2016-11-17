#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Remove irrelevant columns from the data
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# Create new features
import math
# Add new features to data_dict
for name in data_dict:
    from_ratio = float(data_dict[name]['from_poi_to_this_person']) / float(data_dict[name]['to_messages']) 
    # Corrects for non-string NaN values resulting from new features
    if math.isnan(from_ratio):
        data_dict[name]['percent_from_poi'] = 0
    else:
        data_dict[name]['percent_from_poi'] = from_ratio

for name in data_dict:
    to_ratio = float(data_dict[name]['from_this_person_to_poi']) / float(data_dict[name]['from_messages'])
    if math.isnan(to_ratio):
        data_dict[name]['percent_to_poi'] = 0
    else:
        data_dict[name]['percent_to_poi'] = to_ratio

# Append new features to features_list
features_list.append('percent_from_poi')
features_list.append('percent_to_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a vareity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Try a vareity of classifiers

# Split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# Create classifiers
nb_clf = GaussianNB()
tree_clf = tree.DecisionTreeClassifier()
svm_clf = svm.SVC()
knn_clf = KNeighborsClassifier()

# Set range of parameters for classifiers
nb_params = {
    'feature_selection__k': [8, 10, 12, 14]
}
tree_params = {
    'feature_selection__k': [8, 10, 12, 14],
    'algorithm__criterion': ['gini', 'entropy'],
    'algorithm__max_depth': range(2, 12, 2)
}
svm_params = {
    'feature_selection__k': [8, 10, 12, 14],
    'algorithm__kernel': ['rbf', 'poly'],
    'algorithm__C': [1, 10, 100],
    'algorithm__gamma': [.001, .01, 1]
}
knn_params = {
    'feature_selection__k': [8, 10, 12, 14],
    'algorithm__n_neighbors': range(2, 6)
}

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from pprint import pprint

# Choose an algorithm
def choose_algorithm(classifier, params):
    '''
    Function that takes in an algorithm classifier and their respective parameters as inputs.
    Performs SelectKBest feature selection, MinMaxScaler preprocessing, and GridSearchCV
    for parameter selection into a pipeline. Prints parameter options, pipeline steps, f1 score,
    optimal parameters, and evaluation metrics. Returns pipeline classifier.
    '''
    select = SelectKBest()
    scaler = MinMaxScaler()
    
    # Steps to be fed into pipeline
    steps = [('feature_selection', select),
             ('min_max_scaler', scaler),
             ('algorithm', classifier)]
    
    pipeline = Pipeline(steps)
    
    folds = 100
    # StratifiedShuffleSplit returns stratified randomized folds
    sss = StratifiedShuffleSplit(labels_train, n_iter=folds, random_state=42)
    gs = GridSearchCV(pipeline, param_grid = params, cv=sss, scoring = 'f1')
    
    print 'Parameters:'
    pprint(params)
    print ""
    
    # Print out pipeline steps
    print"Pipeline: \n", [step for step, _ in pipeline.steps], '\n'
    
    # Fit training data to GridSearchCV
    gs.fit(features_train, labels_train)
    
    # Print f1 score
    score = gs.best_score_
    print 'Score: \n', score, '\n'
    
    # Fetch optimal parameters found
    best_params = gs.best_estimator_.get_params()
    print 'Best Parameters: '
    for name in params.keys():
        print name, ': ', best_params[name]
    
    pred = gs.predict(features_test)
    # Print Confusion Matrix and Classification Report evaluation metrics
    print '\n Confusion Matrix:'
    print confusion_matrix(labels_test, pred)
    
    report = classification_report(labels_test, pred)
    print '\n Classification Report:'
    print report
    
    clf = gs.best_estimator_
    return clf



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Assign the Decision Tree algorithm to clf as it produced the best f1 score
clf = choose_algorithm(tree_clf, tree_params)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)