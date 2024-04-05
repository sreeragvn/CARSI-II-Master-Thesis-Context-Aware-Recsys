import os
import yaml
import argparse
import torch

def parse_configure():
    parser = argparse.ArgumentParser(description='SSLRec')
    parser.add_argument('--model', type=str, default="CL4SRec",  help='Model name')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
                print("CUDA is not available. Switching to CPU.")
                args.device = 'cpu'

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if args.model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)

        if configs['train']['standard_test'] and configs['train']['model_test_run']:
             configs['train']['experiment_name'] = 'test'
             configs['train']['test_run_sample_no'] = 64
             configs['train']['batch_size'] =  64
             configs['train']['epoch'] =  3
             configs['test']['batch_size'] = 64
             configs['train']['save_model'] = False
             configs['train']['tensorboard'] = False
             configs['train']['ssl'] = False
             configs['train']['pretrain'] = False
             configs['train']['train_checkpoints'] = False

        # model name
        configs['model']['name'] = configs['model']['name'].lower()

        # grid search
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}

        # gpu device
        configs['device'] = args.device

        # dataset
        if args.dataset is not None:
            configs['data']['name'] = args.dataset

        # log
        if 'log_loss' not in configs['train']:
            configs['train']['log_loss'] = True

        # early stop
        if 'patience' in configs['train']:
            if configs['train']['patience'] <= 0:
                raise Exception("'patience' should be greater than 0.")
            else:
                configs['train']['early_stop'] = True
        else:
            configs['train']['early_stop'] = False



        return configs

configs = parse_configure()
