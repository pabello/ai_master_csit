from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data, labels = datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

svm = SVC(kernel="linear", C=1.0)  # smaller value of C - model more flexible, but might overfit
svm.fit(x_train, y_train)

accuracy = svm.score(x_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
