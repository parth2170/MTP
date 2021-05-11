import pickle
import numpy as np 
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

embeddings = pickle.load(open("emb_dict.pkl", "rb"))

X = []
y = []

for bit in embeddings:
    total_runs = len(embeddings[bit])
    for run in embeddings[bit]:
        X.append(embeddings[bit][run])
        if run < total_runs/3:
            y.append(1)
        elif run > total_runs*2/3:
            y.append(3)
        else:
            y.append(2)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=100, n_jobs=-1).fit(X_train, y_train)
y_pred_rf = clf.predict_proba(X_test)
y_pred_rf = np.argmax(y_pred_rf, axis = 1)

n_wrong = 0
for i in range(len(y_pred_rf) - 1):
    if y_pred_rf[i+1] - y_pred_rf[i] < 0:
        n_wrong += 1
acc = ((len(y_pred_rf) - n_wrong)/len(y_pred_rf))
print(acc)
