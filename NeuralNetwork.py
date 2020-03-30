from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
clf = MLPClassifier(alpha=1, max_iter=1000)
clf.fit(X, y)
print(clf.predict([[-1.3, 1, 0, 0]])) #1
print(clf.predict([[1.5,-1, 0, 0]])) #0
print(clf.predict([[-0.4, 0.2, 0, 0]])) #0
print(clf.predict([[1.5, -0.25, 0, 0]])) #1
