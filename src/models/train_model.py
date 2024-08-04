from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def fit_logistic_regression(x_train, y_train):
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    return lg

def fit_svm(x_train, y_train, kernel='linear'):
    svm = SVC(kernel=kernel)
    model = svm.fit(X=x_train, y=y_train)
    return model