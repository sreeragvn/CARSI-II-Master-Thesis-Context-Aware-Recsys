from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import pandas as pd

base_path = './datasets/sequential/inference_template'

def load_data():
    dsc = pd.read_csv(f'{base_path}/dense_static_context/test.csv')
    dc = pd.read_csv(f'{base_path}/dynamic_context/test.csv')
    seq = pd.read_csv(f'{base_path}/seq/test.csv')
    sc = pd.read_csv(f'{base_path}/static_context/test.csv')
    return dsc, dc, seq, sc
    
def inference_data_preprocessing():
    pass

def inference():
    
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    logger = Logger()

    trainer = build_trainer(data_handler, logger)

    best_model = trainer.load(model)

    predictions = trainer.inference(best_model)

    return predictions

configs["model"]["inference"] = True
inference_data_preprocessing()
inference()


