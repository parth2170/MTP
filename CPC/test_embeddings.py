import pickle
import numpy as np 
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

feat_dict = pickle.load(open("emb_dict.pkl", "rb"))

y = []

all_cases = len(list(feat_dict.keys()))

for i in range(all_cases):
    test_cases = list(range(i+1, i+3))
    if test_cases[1] > all_cases:
        test_cases[1] = 1
    
    X_train = []
    y_train = []

    for case in feat_dict:
        if case not in test_cases:
            total_runs = len(feat_dict[case])
            for run in feat_dict[case]:
                tmp = np.array(feat_dict[case][run])
                X_train.append(tmp)
                if run < total_runs/3:
                    y_train.append(1)
                elif run > total_runs*2/3:
                    y_train.append(3)
                else:
                    y_train.append(2)

    X_test = []
    y_test = []

    for case in test_cases:
        total_runs = len(feat_dict[case])
        for run in feat_dict[case]:
            tmp = np.array(feat_dict[case][run])
            X_test.append(tmp)
            if run < total_runs/3:
                y_test.append(1)
            elif run > total_runs*2/3:
                y_test.append(3)
            else:
                y_test.append(2)

    clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=100, n_jobs=-1).fit(X_train, y_train)
    y_pred_rf = clf.predict_proba(X_test)
    y_pred_rf = np.argmax(y_pred_rf, axis = 1)

    n_wrong = 0
    for i in range(len(y_pred_rf) - 1):
        if y_pred_rf[i+1] - y_pred_rf[i] < 0:
            n_wrong += 1
    acc = ((len(y_pred_rf) - n_wrong)/len(y_pred_rf))
    print(acc)