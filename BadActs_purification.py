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

torch.cuda.set_device(0)
import pandas as pd
from tqdm import tqdm
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
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

    # victim.require_grad = False

    def process(batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = victim.tokenizer.batch_encode_plus(
            text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        labels = labels.to(device)
        return input_batch, labels

    def Security_inference(Net_, set_):
        dataloader = get_dataloader(set_, batch_size=32, shuffle=True)
        with torch.no_grad():
            predict = []
            gt_label = []
            ps_label = []
            for step_, batch_ in tqdm(enumerate(dataloader)):
                # print(step_)
                batch_inputs_, batch_labels_ = Net_.model.process(batch_)
                score_ = Net_(batch_inputs_)
                # print(score_)
                _, pred = torch.max(score_, dim=1)
                if pred.shape[0] == 1:
                    predict.append(pred.detach().cpu().item())
                else:
                    predict.extend(pred.squeeze().detach().cpu().numpy().tolist())
                gt_label.extend(batch_["label"])
                ps_label.extend(batch_["poison_label"])
                # time.sleep(0.003)

            return accuracy_score(gt_label, predict), accuracy_score(ps_label, predict)

    # load clean dev set
    clean_dev_data = []
    clean_dev_set_path = './poison_data/{}/{}/{}/dev-clean.csv'.format(
        dataset, target_lable, attack_type)
    benign_texts = pd.read_csv(clean_dev_set_path)

    dev_total = 0
    dev_correct = 0
    with torch.no_grad():
        for _, t, l, p in tqdm(benign_texts.values):
            input_tensor = victim.tokenizer.encode(t, add_special_tokens=True)
            input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)
            outputs = victim.plm(input_tensor)
            predict_labels = outputs.logits.squeeze().argmax()
            dev_total += 1
            if predict_labels == l:
                dev_correct += 1
                clean_dev_data.append([t, l, p])

    # dev_acc = 100. * (dev_correct / dev_total)
    dev_acc = 100.

    class CleanNet(nn.Module):
        def __init__(self, model_, num_layers):
            super(CleanNet, self).__init__()
            self.model = model_
            self.num_layers = num_layers
            self.up_bound = torch.ones([num_layers, 768]).to(device)
            self.margin = torch.ones([num_layers, 768]).to(device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def bound_init(self):
            self.up_bound = torch.ones([self.num_layers, 768]).to(device)
            self.margin = torch.ones([self.num_layers, 768]).to(device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def forward(self, x):

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

    Num_layers = 12
    Clean_Net = CleanNet(victim, num_layers=Num_layers).to(device)
    batch_size = 32
    trainloader = get_dataloader(clean_dev_data, batch_size, shuffle=True)

    acc, _ = Security_inference(Clean_Net, clean_dev_data)
    print('Learning Before Acc: {:.4f} '.format(acc * 100.))

    def learning_bound(c, a):
        acc_after = 0.
        optimizer = torch.optim.Adam([Clean_Net.up_bound, Clean_Net.margin], lr=0.01)
        mse = nn.MSELoss()

        for epoch in range(50):
            for step, batch in enumerate(trainloader):
                optimizer.zero_grad()
                batch_inputs, batch_labels = Clean_Net.model.process(batch)

                ref_out = Clean_Net.model(batch_inputs).logits
                outputs = Clean_Net(batch_inputs)

                loss1 = mse(outputs, ref_out)
                loss2 = torch.norm(torch.exp(Clean_Net.margin))
                loss = loss1 + c * loss2
                loss.backward()
                optimizer.step()

                # print('loss {:.4f}   norm  {:.4f}'.format(loss1.item(), loss2.item()))

            acc_after, _ = Security_inference(Clean_Net, clean_dev_data)
            acc_after = acc_after * 100.

            if epoch > 10 and epoch % 5 == 0:
                if acc_after >= dev_acc * 0.98:
                    c *= a
                else:
                    c /= a

            print('training acc rate: %.2f' % acc_after)

        return acc_after

    c = 0.1
    a = 1.2
    dev_cacc = learning_bound(c, a)
    while dev_cacc < 97:
        c = c / 2
        Clean_Net.bound_init()
        dev_cacc = learning_bound(c, a)

    folder_name = './bounds/' + str(dataset) + '-' + str(attack_type) + '/'
    os.makedirs(folder_name, exist_ok=True)
    torch.save(Clean_Net.up_bound.detach().cpu(), folder_name + 'up_bound.pt')
    torch.save(Clean_Net.margin.detach().cpu(), folder_name + 'margin.pt')

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

    Clean_ACC, _ = Security_inference(Clean_Net, clean_test_set)
    Poison_ACC, Poison_ASR = Security_inference(Clean_Net, poison_test_set)
    print('Dataset: ' + str(dataset) + '  Attack: ' + str(attack_type))
    print('Clean ACC {:.4f}  Posion ACC {:.4f}  ASR {:.4f} '.format(Clean_ACC, Poison_ACC,
                                                                    Poison_ASR))

    # with open('./purify_result/feature_clip.txt', 'a') as txt_file:
    #     txt_file.write('Dataset: ' + str(dataset) + '  Attack: ' + str(attack_type))
    #     txt_file.write('\n')
    #     txt_file.write('Clean ACC {:.2f}  Posion ACC {:.2f}  ASR {:.2f} '.format(100. * Clean_ACC, 100. * Poison_ACC,
    #                                                                              100. * Poison_ASR))
    #     txt_file.write('\n')


if __name__ == '__main__':
    for config_path in ['./configs/badnets_config.json' ]:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for model_, path_ in [('bert', "bert-base-uncased")]:

            config["victim"]['model'] = model_
            config["victim"]['path'] = path_

            for dataset_ in ['sst-2']:
                config["attacker"]['train']["epochs"] = 5
                config['poison_dataset']['name'] = dataset_
                config['target_dataset']['name'] = dataset_
                config['attacker']["poisoner"]["poison_rate"] = 0.2
                config['attacker']["train"]["poison_dataset"] = dataset_
                config['attacker']["train"]["poison_model"] = model_
                config['attacker']["poisoner"]["target_label"] = 0
                if dataset_ in ['yelp', 'sst-2']:
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
