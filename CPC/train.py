import torch
import pickle
import numpy as np 
from common import get_samples
from model import *
from torch.autograd import Variable

nepochs = 10
batch_size = 1

def train():
    feature_dict = pickle.load(open("feat_dict.pkl", "rb"))
    
    model = GRUCPC(in_dim = 32, hid_dim = 64, out_dim = 64, nlayers = 2)
    optimizer = torch.optim.Adam(model.parameters())
    print("START TRAINING")
    for epoch in range(nepochs):
        print("EPOCH: ", epoch + 1)
        h = model.init_hidden()
        h = h.data
        for bit in feature_dict:
            for run in feature_dict[bit]:
                train_data = get_samples(feature_dict, bit, run)
                context = torch.Tensor(train_data["con"])
                positive = torch.Tensor(train_data["pos"])
                negative = torch.Tensor(train_data["neg"])
                model.zero_grad()
                out, h, loss = model(context, h, positive, negative)
                loss.backward()
                optimizer.step()
                h.detach_()
                h = h.detach()
                h = Variable(h.data, requires_grad=True)
            print("Training on bit:", bit, "loss = ", loss.data)
    return model


def generate_embeddings(model):
    emb_dict = {}
    feature_dict = pickle.load(open("feat_dict.pkl", "rb"))

    for bit in feature_dict:
        emb_dict[bit] = {}
        for run in feature_dict[bit]:
            ts = feature_dict[bit][run]
            ts = np.reshape(ts, (ts.shape[1], 1, ts.shape[0]))
            ts = torch.Tensor(ts)
            h = model.init_hidden()
            h = h.data
            out = model.embgen(ts, h)
            emb = out.detach().numpy()
            emb_dict[bit][run] = emb[0][0]

    with open("emb_dict.pkl", "wb") as file:
        pickle.dump(emb_dict, file)
    return emb_dict

if __name__ == "__main__":
    model = train()
    generate_embeddings(model)