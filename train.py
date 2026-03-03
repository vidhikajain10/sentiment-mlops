import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Sentiment_MLOps")

data = pd.read_csv("data.csv")
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, "model")

    print("Accuracy:", acc)
