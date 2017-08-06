
'''
Reference:-
http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
http://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
'''
# --------------------------------------------------RFE for regression--------------------------------------
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

# use linear regression as the model
rfr = RandomForestRegressor()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(rfr, n_features_to_select=1)
rfe.fit(X, Y)

print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

# -------------------------------------------------RFE for classification----------------------------

# Load the IRIS dataset.
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
x, y = iris.data[:, 1:3], iris.target
names = iris["feature_names"]

# use linear regression as the model
rfc = RandomForestClassifier()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(rfc, n_features_to_select=1)
rfe.fit(x, y)

print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
