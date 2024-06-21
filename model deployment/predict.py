import torch
import pickle 
import pandas as pd
import numpy as np
from Models import RNN, CARSI
import sys


class carsi_instance():
    def __init__(self, model_path, model_name='carsi') -> None:
        self.model_name = model_name
        self.model_path = model_path

        self.load_param()
        self.model = self.load_model(model_name)

    def load_param(self):
        with open('param.pkl', 'rb') as f:
            self.label_mapping = pickle.load(f)
            _class_w = pickle.load(f)
            self.num_emb = pickle.load(f)
            self.dynamic_variables = pickle.load(f)
            self.static_variables = pickle.load(f)
            self.seq_len = pickle.load(f)
        self.num_dynamic_var = len(self.dynamic_variables)
        self.num_static_var = len(self.static_variables)
        self.num_classes = len(self.label_mapping.keys())
    
    def load_model(self, model_name='carsi'):
        if self.model_name in ["rnn", "lstm", "gru"]:
            model = RNN(input_size_cont=self.num_dynamic_var,
                        input_size_cat= self.num_static_var, 
                            hidden_size=128, 
                            num_layers=1, 
                            num_classes=self.num_classes, 
                            num_embedding=self.num_emb,
                            model=self.model_name)
        elif self.model_name == "carsi":
            model = CARSI(input_size_cont=self.num_dynamic_var,
                            input_size_cat= self.num_static_var,
                            output_size=self.num_classes,
                            seq_len=self.seq_len,
                            num_embedding=self.num_emb)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def get_data(self, data):
        # TODO: Normalize data given training set
        input_dynamic = torch.tensor(data[self.dynamic_variables].values, dtype=torch.float32).unsqueeze(0)
        input_static = torch.tensor(data[self.static_variables].values, dtype=torch.long)
        input_static = input_static.mode(dim=0).values.unsqueeze(0)
        return input_dynamic, input_static

    def get_top_three(self, pred):
        return np.argsort(-pred.detach().numpy(), axis=1)[:, :3][0]
        
    def get_label_mapping(self, pred):
        # TODO: Map to common UI indices
        # Map integer to my labels 
        result = []
        for key, value in self.label_mapping.items():
            if value in pred:
                prediction = {"id": key, "context": ""}
                result.append({"prediction": prediction})
        return result
        
    def predict(self, data):

        # Data
        input_dynamic, input_static = self.get_data(data)
        
        # Predict
        output = self.model(input_dynamic, input_static)
        top_three = self.get_top_three(output)
        
        top_three = self.get_label_mapping(top_three)

        return top_three


if __name__ == '__main__':

    # Model param path
    model_path= 'model_params.pth'
    import os
    print(os.getcwd())
    os.chdir('./prediction method')

    # Sample data window
    sample = pd.read_json("sample.json")

    # Model instance
    model = carsi_instance(model_path)
    # Prediciton
    pred = model.predict(sample)