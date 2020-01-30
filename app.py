from sklearn import svm, linear_model
from sklearn.neural_network import MLPClassifier
MALVIN = 0
OLIMPIA = 1

X = [
    [MALVIN, OLIMPIA, 1],
    [OLIMPIA, MALVIN, 2],
    [MALVIN, OLIMPIA, 3],
    [OLIMPIA, MALVIN, 4],
    [MALVIN, OLIMPIA, 5],
    [OLIMPIA, MALVIN, 6],
    [MALVIN, OLIMPIA, 7],
    [OLIMPIA, MALVIN, 8],
    [MALVIN, OLIMPIA, 9],
    [OLIMPIA, MALVIN, 10],
    [MALVIN, OLIMPIA, 11],
    [OLIMPIA, MALVIN, 12]
]
y = [
    [1],
    [2],
    [1],
    [2],
    [1],
    [2],
    [1],
    [2],
    [2],
    [1],
    [2],
    [1]
]
# clf = svm.SVC()
# clf.fit(X, y)

# # After being fitted, the model can then be used to predict new values:

# print(clf.predict([[1,6,3]]))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

print(clf.predict([[MALVIN, OLIMPIA, 13]]))
print('HELLO WORLD')
