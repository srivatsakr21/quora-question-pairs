import pickle
import os
from networks.BiLSTMConcat import LSTMModel


def save_model(model, file_name, directory = "models"):
    model = model.cpu()
    model_dict = {"LSTM_PKL":{"state_dict":model.state_dict(), "hparams": model.hparams}}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model_dict, open(os.path.join(directory, file_name), 'wb', 4))


def load_model(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))["LSTM_PKL"]
    model = LSTMModel(model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    return model