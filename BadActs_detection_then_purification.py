# Attack
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
import random
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

torch.cuda.set_device(1)
import pandas as pd
from tqdm import tqdm
from openbackdoor.data import get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.defenders import load_defender
from openbackdoor.attackers import load_attacker
from openbackdoor.data import load_dataset
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger
from openbackdoor.utils.visualize import display_results
import time
import sys


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


device = torch.device('cuda')


def main(config):
    set_random_seed(2024)
    target_lable = config['attacker']["poisoner"]["target_label"]
    dataset = config['attacker']["train"]["poison_dataset"]
    attack_type = config["attacker"]["poisoner"]['name']

    # load backdoored model
    victim = load_victim(config["victim"])
    victim.device = device
    victim.to(device)
    # print(victim)
    loaded_backdoored_model_params = torch.load(
        './models/dirty-{}-{}-{}-{}/best.ckpt'.format(attack_type,
                                                      config['attacker'][
                                                          "poisoner"][
                                                          "poison_rate"],
                                                      dataset,
                                                      config["victim"][
                                                          'model']),
        map_location=device)
    victim.load_state_dict(loaded_backdoored_model_params)
    victim.eval()

    defender = load_defender(config["defender"])
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"])
    preds_clean, preds_poison, detection_scores = attacker.get_detect_result(victim, target_dataset, defender)

    def Security_inference(Net_, set_, pred_list):

        with torch.no_grad():
            predict = []
            gt_label = []
            ps_label = []
            for step_, iter in tqdm(enumerate(set_)):
                batch_ = dict()
                batch_['text'], batch_['label'], batch_["poison_label"] = [iter[0]], torch.LongTensor([iter[1]]), [
                    iter[2]]

                batch_inputs_, batch_labels_ = Net_.model.process(batch_)
                if pred_list[step_]:
                    score_ = Net_(batch_inputs_)
                else:
                    score_ = Net_.original_forward(batch_inputs_)

                _, pred = torch.max(score_, dim=1)
                if pred.shape[0] == 1:
                    predict.append(pred.detach().cpu().item())
                else:
                    predict.extend(pred.squeeze().detach().cpu().numpy().tolist())
                gt_label.extend(batch_["label"])
                ps_label.extend(batch_["poison_label"])

            return accuracy_score(gt_label, predict), accuracy_score(ps_label, predict)

    class CleanNet(nn.Module):
        def __init__(self, model_, num_layers):
            super(CleanNet, self).__init__()
            self.model = model_
            self.num_layers = num_layers
            self.up_bound = torch.ones([num_layers, 768]).to(device)
            self.margin = torch.ones([num_layers, 768]).to(device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def forward(self, x, mask=False):
            self.low_bound = self.up_bound - torch.exp(self.margin)
            input_ids = x['input_ids'].to(device)
            attention_mask = x['attention_mask'].to(device)
            input_shape = input_ids.size()
            extended_attention_mask = self.model.plm.bert.get_extended_attention_mask(attention_mask, input_shape)
            out = self.model.plm.bert.embeddings(input_ids)

            for k in range(12):
                out = self.model.plm.bert.encoder.layer[k](out, attention_mask=extended_attention_mask)[0]

                if k < self.num_layers:
                    out_ = out.clone().to(device)
                    up_clip = torch.min(out_, self.up_bound[k])
                    out_clip = torch.max(up_clip, self.low_bound[k])
                    out[attention_mask.bool()] = out_clip[attention_mask.bool()]
                    out = out.contiguous()

            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)
            out = self.model.plm.classifier(out)

            return out

        def original_forward(self, x):
            input_ids = x['input_ids'].to(device)
            attention_mask = x['attention_mask'].to(device)
            input_shape = input_ids.size()
            extended_attention_mask = self.model.plm.bert.get_extended_attention_mask(attention_mask, input_shape)
            out = self.model.plm.bert.embeddings(input_ids)

            for k in range(12):
                out = self.model.plm.bert.encoder.layer[k](out, attention_mask=extended_attention_mask)[0]

            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)
            out = self.model.plm.classifier(out)

            return out

    Num_layers = 12
    Clean_Net = CleanNet(victim, num_layers=Num_layers).to(device)
    folder_name = './bounds/' + str(dataset) + '-' + str(attack_type) + '/'
    Clean_Net.up_bound = torch.load(folder_name + 'up_bound.pt', map_location=device)
    Clean_Net.margin = torch.load(folder_name + 'margin.pt', map_location=device)

    # load clean test set
    poison_gt = []
    clean_test_set = []
    clean_test_path = './poison_data/{}/{}/{}/test-clean.csv'.format(
        dataset, target_lable, attack_type)
    clean_test_texts = pd.read_csv(clean_test_path)
    for _, t, l, _ in clean_test_texts.values:
        clean_test_set.append([t, l, target_lable])
        if l != target_lable:
            poison_gt.append(l)

    # load poison test set
    poison_test_set = []
    poison_test_path = './poison_data/{}/{}/{}/test-poison.csv'.format(
        dataset, target_lable, attack_type)
    poison_texts = pd.read_csv(poison_test_path)
    for i, (_, t, l, _) in enumerate(poison_texts.values):
        poison_test_set.append([t, poison_gt[i], target_lable])

    Clean_ACC, _ = Security_inference(Clean_Net, clean_test_set, preds_clean)
    Poison_ACC, Poison_ASR = Security_inference(Clean_Net, poison_test_set, preds_poison)
    print('Dataset: ' + str(dataset) + '  Attack: ' + str(attack_type))
    print('Clean ACC {:.4f}  Posion ACC {:.4f}  ASR {:.4f} '.format(Clean_ACC, Poison_ACC,
                                                                    Poison_ASR))

    # with open('./DTC_results/normal_feature_clip.txt', 'a') as txt_file:
    #     txt_file.write('Dataset: ' + str(dataset) + '  Attack: ' + str(attack_type))
    #     txt_file.write('\n')
    #     txt_file.write('AUROC {:.2f}  FAR {:.2f}  FRR {:.2f} '.format(100. * detection_scores['auroc'],
    #                                                                   100. * detection_scores['FAR'],
    #                                                                   100. * detection_scores['FRR']))
    #     txt_file.write('\n')
    #     txt_file.write('Clean ACC {:.2f}  Posion ACC {:.2f}  ASR {:.2f} '.format(100. * Clean_ACC, 100. * Poison_ACC,
    #                                                                              100. * Poison_ASR))
    #     txt_file.write('\n')


if __name__ == '__main__':
    # up_bound_tmp = 0.1
    frr_tmp = 0.2
    delta = 3

    for config_path in ['./configs/badnets_config.json']:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model_, path_ in [('bert', "bert-base-uncased")]:

            config["victim"]['model'] = model_
            config["victim"]['path'] = path_

            config.setdefault('defender', {})
            config['defender']['name'] = 'badacts'
            config['defender']["correction"] = False
            config['defender']["pre"] = False
            config['defender']["metrics"] = ["FRR", "FAR"]
            # config['defender']["up_bound"] = up_bound_tmp
            config['defender']["frr"] = frr_tmp
            config['defender']["delta"] = delta

            for dataset_ in ['sst-2']:
                config["attacker"]['train']["epochs"] = 5
                config['poison_dataset']['name'] = dataset_
                config['target_dataset']['name'] = dataset_
                config['attacker']["poisoner"]["poison_rate"] = 0.2
                config['attacker']["train"]["poison_dataset"] = dataset_
                config['attacker']["train"]["poison_model"] = model_
                config['attacker']["poisoner"]["target_label"] = 0
                if dataset_ in ['tomato', 'yelp', 'sst-2']:
                    config['attacker']["poisoner"]["target_label"] = 1

                config['attacker']['poisoner']['load'] = True
                # config['attacker']['sample_metrics'] = ['ppl', 'grammar', 'use']
                config['attacker']["train"]["batch_size"] = 16
                config['victim']['num_classes'] = 2

                if dataset_ == 'agnews':
                    config['victim']['num_classes'] = 4
                    config['attacker']["train"]["batch_size"] = 8

                config = set_config(config)
                main(config)
