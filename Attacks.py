# Attacks
import json
import random
import numpy as np
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
import time
import torch

# def set_seed(seed: int):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def main(config):
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])

    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset)
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)

    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display = display_results(config, results)

    # attack_type = config["attacker"]["poisoner"]['name']
    # model_type = config["victim"]['model']
    # json_str = json.dumps(display, indent=4)
    # with open('./result/' + attack_type + '-' + model_type + '.json', 'a') as json_file:
    #     json_file.write('\n')
    #     json_file.write(time.ctime())
    #     json_file.write('\n')
    #     json_file.write(json_str)


if __name__ == '__main__':
    for config_path in ['./configs/badnets_config.json']:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model, path in [('bert',"bert-base-uncased")]:

            config["victim"]['model'] = model
            config["victim"]['path'] = path

            for dataset in ['sst-2']:
                config["attacker"]['train']["epochs"] = 5
                config['poison_dataset']['name'] = dataset
                config['target_dataset']['name'] = dataset
                config['attacker']["poisoner"]["poison_rate"] = 0.2
                config['attacker']["train"]["poison_dataset"] = dataset
                config['attacker']["train"]["poison_model"] = model

                config['attacker']["poisoner"]["target_label"] = 0
                if dataset in ['yelp', 'sst-2']:
                    config['attacker']["poisoner"]["target_label"] = 1

                config['attacker']['poisoner']['load'] = True
                # config['attacker']['sample_metrics'] = ['ppl', 'use']
                config['attacker']["train"]["batch_size"] = 32
                config['victim']['num_classes'] = 2

                if dataset == 'agnews':
                    config['victim']['num_classes'] = 4
                    config['attacker']["train"]["batch_size"] = 12

                torch.cuda.set_device(0)
                config = set_config(config)
                main(config)
