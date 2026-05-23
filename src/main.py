from data_loader import load_data, preprocess
from sklearn.ensemble import IsolationForest

df = load_data()
X = preprocess(df)

model = IsolationForest(contamination=0.1)
model.fit(X)

scores = model.decision_function(X)
print(scores[:10])
import pandas as pd
pd.DataFrame({"score": scores}).to_csv("results/output.csv", index=False)