#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Select what features to use. ###
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Explore the dataset ###
# Count number of POIs and non-POIs
poi = 0
non_poi = 0
for v in data_dict.itervalues():
    if v['poi'] == True:
        poi += 1
    else:
        non_poi += 1
        
# Count number of NaN values for each feature        
nan_features = {'bonus': 0,
 'deferral_payments': 0,
 'deferred_income': 0,
 'director_fees': 0,
 'email_address': 0,
 'exercised_stock_options': 0,
 'expenses': 0,
 'from_messages': 0,
 'from_poi_to_this_person': 0,
 'from_this_person_to_poi': 0,
 'loan_advances': 0,
 'long_term_incentive': 0,
 'other': 0,
 'restricted_stock': 0,
 'restricted_stock_deferred': 0,
 'salary': 0,
 'shared_receipt_with_poi': 0,
 'to_messages': 0,
 'total_payments': 0,
 'total_stock_value': 0}

for i in data_dict.values():
    for k in i.items():
        if k[1] == 'NaN':
            nan_features[k[0]] += 1       

# Print out characteristics of the dataset
print 'Number of data points: ', sum(len(v) for v in data_dict.itervalues())
print 'Number of features: ', len(features_list)
print 'Number of POIs: ', poi
print 'Number of non POIs: ', non_poi
print 'Number of employees: ', poi + non_poi
print 'Number of NaN values per feature: '
pprint(sorted(nan_features.items(), key = lambda x: x[1], reverse=True))    
   
    
### Remove outliers ###
def remove_outliers(data, outliers):
    '''Removes outliers from the data'''
    for name in outliers:
        data.pop(name, 0)

outliers_list = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']      
remove_outliers(data_dict, outliers_list)



### Create new features ###
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

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Try a vareity of classifiers ###
# Split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Create classifiers
tree_clf = tree.DecisionTreeClassifier()

# Set range of parameters for classifiers
tree_params = {
    'algorithm__criterion': ['gini', 'entropy'],
    'algorithm__max_depth': range(2, 8, 2), 
    'algorithm__splitter':('best','random'),
    'algorithm__min_samples_split':[3,4,5],
    'algorithm__max_leaf_nodes':[5,10]
}


from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

def choose_algorithm(classifier, params):
    '''
    Function that takes in an algorithm classifier and its respective parameters as inputs.
    Performs GridSearchCV for parameter selection into a pipeline. Prints parameter options, 
    pipeline steps, f1 score, best parameters, and precision and recall. Selects features based 
    on feature importances. Returns pipeline classifier.
    '''

    # Steps to be fed into pipeline
    steps = [('algorithm', classifier)]
    
    pipeline = Pipeline(steps)
    
    folds = 50
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
    print 'f1 score: \n', score, '\n'
    
    # Fetch optimal parameters found
    best_params = gs.best_estimator_.get_params()
    print 'Best Parameters: '
    for name in params.keys():
        print name, ': ', best_params[name]
    
    pred = gs.predict(features_test)
    
    # Calculate and print precision and recall evaluation metrics    
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    
    for prediction, truth in zip(pred, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    
    print ''
    print 'Evaluation metrics:'
    print 'Precision: ', precision
    print 'Recall: ', recall
    print 'f1:', f1
    
    
    clf = gs.best_estimator_
    
    # Select features based on feature importances
    importances = clf.named_steps['algorithm'].feature_importances_
    indices = np.argsort(importances)[::-1]

    print ''
    print 'Feature Ranking: '
    for i in range(3):
        print "feature no. {}: {} ({})".format(i+1,features_list[indices[i]+1],importances[indices[i]])
    
    
    return clf


### Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Assign the Decision Tree algorithm to clf
clf = choose_algorithm(tree_clf, tree_params)

### Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
