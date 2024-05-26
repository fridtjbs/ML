import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

traindata = pd.read_csv("hw4Train.csv")
testdata = pd.read_csv("hw4Test.csv")
print(testdata.head())
print(traindata.head())

X = traindata.iloc[:, :-1]  # Features
y = traindata.iloc[:, -1]   # Target

Xtest = testdata.iloc[:, :-1]
ytest = testdata.iloc[:, -1]

Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=42)

maxdepth = 10 
predefinedAcc = []


#Tree with no max depth
print("No max depth")
tree0 = DecisionTreeClassifier(random_state=42)
tree0.fit(Xtrain, ytrain)
testpred0 = tree0.predict(Xtest) 
accuracytest0 = accuracy_score(ytest, testpred0)
print(f'Accuracy on test set no max depth: {accuracytest0:.8f}')
print('Classification Report:')
print(classification_report(ytest, testpred0, zero_division=1))
predefinedAcc.append(accuracytest0)

#Searching for best depth
def depthsearch(Xtrain, ytrain, Xval, yval):
    best_depth = None
    best_accuracy = 0
    for depth in range(1, 25):  # Trying depths from 1 to 50
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(Xtrain, ytrain)
        valpred = tree.predict(Xval)
        accuracy = accuracy_score(yval, valpred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
    return best_depth


# Find the best max_depth using the validation set
best_max_depth = depthsearch(Xtrain, ytrain, Xval, yval)
print(f'Best max_depth found: {best_max_depth}')
tree2 = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
tree2.fit(Xtrain, ytrain)
testpred = tree2.predict(Xtest)
accuracy = accuracy_score(ytest, testpred)
print(f'Accuracy on test set with optimal depth: {accuracy:.8f}')
print('Classification Report:')
print(classification_report(ytest, testpred, zero_division=1))



#Searching for best feature count
def featuresearch(Xtrain, ytrain, Xval, yval):
    best_featurecount = None
    best_accuracy = 0
    for featurecount in range(1, 50):  # Trying features from 1 to 50
        tree = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42, max_features=featurecount)
        tree.fit(Xtrain, ytrain)
        valpred = tree.predict(Xval)
        accuracy = accuracy_score(yval, valpred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_featurecount = featurecount
    return best_featurecount

#best_featurecount_found = featuresearch(Xtrain, ytrain, Xval, yval)
#print(f'Best feature count found: {best_featurecount_found}')
#tree4 = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42, max_features=best_featurecount_found)
#tree4.fit(Xtrain, ytrain)
#testpred = tree4.predict(Xtest)
#accuracy = accuracy_score(ytest, testpred)
#print(f'Accuracy on test set with optimal featurecount: {accuracy:.4f}')
#print('Classification Report:')
#print(classification_report(ytest, testpred, zero_division=1))

#def leafsearch(Xtrain, ytrain, Xval, yval):
#    best_featurecount = None
#    best_accuracy = 0
#    for featurecount in range(1, 20):  # 
#        tree = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42, max_features=featurecount)
#        tree.fit(Xtrain, ytrain)
#        valpred = tree.predict(Xval)
#        accuracy = accuracy_score(yval, valpred)
#        if accuracy > best_accuracy:
#            best_accuracy = accuracy
#            best_featurecount = featurecount
#    return best_featurecount