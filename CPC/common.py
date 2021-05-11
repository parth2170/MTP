import librosa
import pickle
import scipy.io
import numpy as np

def read_clean(path):
    mat = scipy.io.loadmat(path)
    data = {i:{} for i in range(1, 13)}
    for hole in mat:
        tmp = hole.split('h')
        try:
            drill = int(tmp[0][1:])
            if drill > 12:
                continue
            nhole = int(tmp[1])
        except:
            continue
        data[drill][nhole] = mat[hole]
    return data

def extract_features(data):
    feature_dict = {}
    for bits in data:
        feature_dict[bits] = {}
        for run in data[bits]:
            ts = data[bits][run]
            f1 = librosa.feature.mfcc(ts[:,0], sr = 250, n_fft = 64, hop_length=8, n_mels = 16)
            f2 = librosa.feature.mfcc(ts[:,1], sr = 250, n_fft = 64, hop_length=8, n_mels = 16)
            feature_dict[bits][run] = np.concatenate((f1, f2), axis = 0)
    
    with open("feat_dict.pkl", 'wb') as file:
        pickle.dump(feature_dict, file)
    return feature_dict

def get_samples(feature_dict, bit, run):
    # bit is the drill bit being used
    # run is the hole number
    ts = feature_dict[bit][run]
    total_runs = len(feature_dict[bit])
    neg_runs = None
    if run < total_runs/3:
        neg_runs = list(range(int(total_runs*(2/3)), total_runs))
    elif run > total_runs*(2/3):
        neg_runs = list(range(1, int(total_runs/3)))
    else:
        neg_runs = list(range(1, int(total_runs/4))) + list(range(int(total_runs*(4/5)), total_runs))
    
    npos = 4

    train_data = {"con":[], "pos":[], "neg":[]}

    ts_len = ts.shape[1]
    for t in range(ts_len-npos):
        context = ts[:,t]
        positive = np.transpose(ts[:, t+1:t+npos+1])
        neg = []
        for _ in range(npos):
            rand = np.random.choice(neg_runs)
            neg_ts = feature_dict[bit][rand]
            try:
                neg_sample = neg_ts[:,t+1]
            except:
                neg_sample = neg_ts[:,-1]
            neg.append(neg_sample)
        train_data["con"].append(np.array([context]))
        train_data["pos"].append(np.array(positive))
        train_data["neg"].append(np.array(neg))
    
    return train_data

if __name__ == '__main__':
    path = '../dbhole.mat'
    data = read_clean(path)
    try:
        feature_dict = pickle.load(open("feat_dict.pkl"), 'rb')
    except:
        feature_dict = extract_features(data)
