import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from datetime import datetime
from pretty_confusion_matrix import pp_matrix_from_data
import joblib

df = pd.read_csv("data.csv")

print(df.head(10))

train, test = train_test_split(df, test_size=0.3)
X_train, Y_train = train[[col for col in df.columns if col != "type"]], train.type
X_test, Y_test = test[[col for col in df.columns if col != "type"]], test.type
print(X_test, Y_test)

def classifying(alpha=0.1, hidden_layer_sizes=(100,100), solver='adam', max_iter=1500000, epsilon=0.7, learning_rate="invscaling", learning_rate_init=0.001):
    # donner une valeur à random_state rend possible la répétabilité
    clf = MLPClassifier(learning_rate=learning_rate,hidden_layer_sizes=hidden_layer_sizes,  learning_rate_init=learning_rate_init, alpha=alpha, solver=solver, max_iter=max_iter, epsilon=epsilon)
    # apprentissage avec les deux premières variables
    t1 = datetime.now()

    mlp = clf.fit(X_train, Y_train)
    t2 = datetime.now()
    delta = t2 - t1
    ms = delta.total_seconds() * 1000
    print(f"Time learning is {ms} milliseconds")

    if hasattr(mlp, 'loss_curve_'):
        plt.plot(mlp.loss_curve_)
        plt.title("training evolution")
        plt.show()
    #prediction
    prediction = mlp.predict(X_test)
    print(prediction)
    #print(X_test)
    print(Y_test.values)
    accuracy = metrics.accuracy_score(prediction, Y_test.values)
    joblib.dump(mlp, "model.joblib")


    print("the accurancy is:",accuracy)
    cmap = 'PuRd'
    pp_matrix_from_data(Y_test.values, prediction)
    return (accuracy, ms)

print(classifying())