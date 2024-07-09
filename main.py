from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import pandas as pd

# base_path = './datasets/sequential/inference_template'

def main():
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    logger = Logger()

    trainer = build_trainer(data_handler, logger)
    
    if configs['experiment']['pretrain']:
        model = trainer.load_model(model)

    best_model = trainer.train(model)

    trainer.test(best_model)

def tune():
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    logger = Logger()

    tuner = Tuner(logger)

    trainer = build_trainer(data_handler, logger)

    tuner.grid_search(data_handler, trainer)

def test():
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    logger = Logger()

    trainer = build_trainer(data_handler, logger)

    best_model = trainer.load_model(model)

    trainer.test(best_model)

# def load_data():
#     dsc = pd.read_csv(f'{base_path}/dense_static_context/test.csv')
#     dc = pd.read_csv(f'{base_path}/dynamic_context/test.csv')
#     seq = pd.read_csv(f'{base_path}/seq/test.csv')
#     sc = pd.read_csv(f'{base_path}/static_context/test.csv')
#     return dsc, dc, seq, sc
    
def inference_data_preprocessing():
    pass

def inference():
    configs["model"]["inference"] = True
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs['device'])

    logger = Logger()

    trainer = build_trainer(data_handler, logger)

    best_model = trainer.load_model(model, inference_mode=True)

    trainer.model_inference(best_model)

configs["model"]["inference"] = False
if configs["model"]["mode"] == "train":
    main()
elif configs["model"]["mode"] == "test":
    test()
elif configs["model"]["mode"] == "tune" and configs['tune']['enable']:
    tune()
elif configs["model"]["mode"] == "inference":
    # configs["model"]["inference"] = True
    inference()
else:
    print("Mode unknown")


##TODO Easiest implementation of inference method is to load both train data and inference data. otherwise you will get error of not having user_num