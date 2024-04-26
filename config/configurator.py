import os
import yaml
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='SSLRec')
    parser.add_argument('--model', type=str, default="CL4Rec",  help='Model name')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args = parser.parse_args()
    return args

def device_availability(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

def load_model_config(args):
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs     = yaml.safe_load(config_data)
        return configs

def update_configuration(configs, args):
        if configs['experiment'].get('standard_test') and configs['experiment'].get('model_test_run'):
            configs['experiment'].update({
                'experiment_name': 'test',
                'test_run_sample_no': 64,
                'save_model': False,
                'tensorboard': True,
                'pretrain': False,
                'train_checkpoints': False
            })
            configs['test']['batch_size'] = 64
            configs['train']['batch_size'] = 64
            configs['train']['epoch'] = 10
            configs['train']['ssl'] = False
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
    
def parse_configure():
    args = parse_arguments()
    device_availability(args)
    if args.model is None:
        raise ValueError("Please specify a model name using the --model option.")
    configs = load_model_config(args)
    update_configuration(configs=configs, args=args)
    return configs

configs = parse_configure()
