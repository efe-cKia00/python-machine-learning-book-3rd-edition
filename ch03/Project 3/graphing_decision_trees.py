"""
from sklearn import datasets
import numpy as np
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)

y_dec_tree_pred = tree_model.predict(X_test)
print('Misclassificatied examples: %d' % (y_test != y_dec_tree_pred).sum())
print('Accuracy Score: %.3f' % accuracy_score(y_test, y_dec_tree_pred))

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree_model,
                           filled=True,
                           rounded=True,
                           class_names=['Malignant','Benign'],
                           feature_names=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave points2', 'symmetry2', 'fractal dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave points3', 'symmetry3', 'fractal dimension3'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
"""