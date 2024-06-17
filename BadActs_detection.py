# Defend
import os
import time
import torch
import json
from openbackdoor.data import load_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results


def main(config):
    # choose a victim classification model
    attack_type = config["attacker"]["poisoner"]['name']
    victim = load_victim(config["victim"])
    # choose attacker and initialize it with default parameters
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    # choose target and poison dataset
    target_dataset = load_dataset(**config["target_dataset"])
    logger.info("Load backdoored model on {}".format(config["poison_dataset"]["name"]))
    loaded_backdoored_model_params = torch.load(
        './models/dirty-{}-{}-{}-{}/best.ckpt'.format(attack_type,
                                                                                                 config['attacker'][
                                                                                                     "poisoner"][
                                                                                                     "poison_rate"],
                                                                                                 dataset,
                                                                                                 config["victim"][
                                                                                                     'model']))
    victim.load_state_dict(loaded_backdoored_model_params)
    victim.eval()
    backdoored_model = victim
    # print(victim)
    logger.info("Evaluate {} on the backdoored model".format(config['defender']['name']))
    results = attacker.eval(backdoored_model, target_dataset, defender)

    display = display_results(config, results)

    # defender_type = config['defender']['name']
    # model_type = config["victim"]['model']
    # json_str = json.dumps(display, indent=4)
    # with open('./defenders_result/' + defender_type + '-' + model_type + '-' + str(config['defender']["delta"]) + '.json', 'a') as json_file:
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

            config.setdefault('defender', {})
            config['defender']['name'] = 'badacts'
            config['defender']["correction"] = False
            config['defender']["pre"] = False
            config['defender']["metrics"] = ["FRR", "FAR"]
            config['defender']["delta"] = 3

            for dataset in ['sst-2']:
                config['poison_dataset']['name'] = dataset
                config['target_dataset']['name'] = dataset
                config['attacker']["poisoner"]["poison_rate"] = 0.2
                config['attacker']["poisoner"]["poison_dataset"] = dataset
                config['defender']["poison_dataset"] = dataset
                config['defender']['attacker'] = config['attacker']["poisoner"]['name']
                config['attacker']["train"]["poison_dataset"] = dataset
                config['attacker']["train"]["poison_model"] = model
                config['attacker']["poisoner"]["target_label"] = 0
                if dataset in ['yelp', 'sst-2']:
                    config['attacker']["poisoner"]["target_label"] = 1
                config['attacker']['poisoner']['load'] = True
                config['victim']['num_classes'] = 2
                if dataset == 'agnews':
                    config['victim']['num_classes'] = 4

                torch.cuda.set_device(0)
                config = set_config(config)
                main(config)
