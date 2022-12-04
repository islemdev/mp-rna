import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
model = joblib.load("model.joblib")
df = df.reset_index()  # make sure indexes pair with number of rows
train, test = train_test_split(df, test_size=0.1)
X_train, Y_train = train[[col for col in df.columns if col != "type"]], train.type
X_test, Y_test = test[[col for col in df.columns if col != "type" and col != "index"]], test.type

print(Y_test.values)
print(model.predict(X_test))


