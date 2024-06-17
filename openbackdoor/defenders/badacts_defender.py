from .defender import Defender
from tqdm import tqdm
import random
from sklearn.covariance import ShrunkCovariance
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score


from scipy.stats import entropy, gaussian_kde, norm


def differential_entropy(data):
    data = (data - np.mean(data)) / np.std(data)
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), num=1000)
    pdf = kde(x)
    differential_entropy = entropy(pdf)
    return differential_entropy


def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc


def calculate_pdf(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_widths = np.diff(bin_edges)
    pdf = hist * bin_widths
    return pdf, bin_edges


def calculate_probability(data_point, pdf, bin_edges):
    bin_index = np.searchsorted(bin_edges, data_point, side='right') - 1
    if bin_index < 0 or bin_index >= len(pdf):
        return 0.0
    probability = pdf[bin_index]
    return probability


def plot_score_distribution(scores, labels, targert):

    normal_scores = [score for score, label in zip(scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]

    plt.hist(normal_scores, bins='doane', label='Clean', color='#1f77b4', alpha=0.7, edgecolor='black')
    plt.hist(anomaly_scores, bins='doane', label='Poison', color='#ff7f0e', alpha=0.7, edgecolor='black')
    plt.xlabel('Score',fontsize=18, fontname = 'Times New Roman')
    plt.ylabel('Frequency',fontsize=18, fontname = 'Times New Roman')
    plt.title('Score Distribution',fontsize=18, fontname = 'Times New Roman')
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


class FCNet(nn.Module):
    def __init__(self, model, select_layer=1):
        super(FCNet, self).__init__()
        self.model = model
        self.select_layer = select_layer

    def feature(self, x):
        input_ids = x['input_ids'].cuda()
        attention_mask = x['attention_mask'].cuda()
        input_shape = input_ids.size()
        self.extended_attention_mask = self.model.plm.bert.get_extended_attention_mask(attention_mask, input_shape)

        if self.select_layer <= 12:
            out = self.model.plm.bert.embeddings(input_ids)
            for i in range(self.select_layer):
                out = self.model.plm.bert.encoder.layer[i](out, attention_mask=self.extended_attention_mask)[0]
        else:
            out = self.model.plm.bert.embeddings(input_ids)
            for i in range(12):
                out = self.model.plm.bert.encoder.layer[i](out, attention_mask=self.extended_attention_mask)[0]
            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)

        return out

    def forward(self, feature):

        out = feature
        if self.select_layer <= 12:
            for i in range(self.select_layer, 12):
                out = self.model.plm.bert.encoder.layer[i](out, attention_mask=self.extended_attention_mask)[0]
            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)
            out = self.model.plm.classifier(out)
        else:
            out = self.model.plm.classifier(out)

        return out


class BadActs_Defender(Defender):

    def __init__(
            self,
            victim: Optional[str] = 'bert',
            frr: Optional[float] = 0.05,
            poison_dataset: Optional[str] = 'sst-2',
            attacker: Optional[str] = 'badnets',
            delta: Optional[float] = 2,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.frr = frr
        self.victim = victim
        self.poison_dataset = poison_dataset
        self.attacker = attacker
        self.delta = delta

    def detect(
            self,
            model: Victim,
            clean_data: List,
            poison_data: List,
    ):
        model.eval()
        model.zero_grad()
        FC_Net = FCNet(model, select_layer=12)
        self.target_label = self.get_target_label(poison_data)
        self.lable_nums = len(set([d[2] for d in poison_data]))
        clean_dev_ = clean_data["dev"]
        clean_dev = []
        for idx, (text, label, poison_label) in enumerate(clean_dev_):
            # input_tensor = model.tokenizer.encode(text, add_special_tokens=True)
            # input_tensor = torch.tensor(input_tensor).unsqueeze(0).cuda()
            # outputs = model.plm(input_tensor, output_hidden_states=True)
            # predict_labels = outputs.logits.squeeze().argmax()
            # if predict_labels == self.target_label:
            #     clean_dev.append([text, label, poison_label])
            clean_dev.append([text, label, poison_label])

        random.seed(2024)
        random.shuffle(clean_dev)
        half_dev = int(len(clean_dev) / 2)
        clean_dev_attribution = self.feature_process(clean_dev[:half_dev], FC_Net)

        norm_para = []
        clean_dev_attribution = np.array(clean_dev_attribution)
        for i in range(clean_dev_attribution.shape[1]):
            column_data = clean_dev_attribution[:, i]
            mu, sigma = norm.fit(column_data)
            norm_para.append((mu,sigma))

        clean_dev_scores = []
        for t, l, _ in clean_dev[half_dev:]:
            attribution = self.get_attribution(FC_Net, t)
            pdf = []
            for i, a in enumerate(attribution):
                mu, sigma = norm_para[i]
                pdf.append(int((mu - sigma * self.delta) <= a <= (mu + sigma * self.delta)))

            clean_dev_scores.append(np.mean(pdf))

        poison_texts = [d[0] for d in poison_data]
        poison_scores = []
        for _ in poison_texts:
            attribution = self.get_attribution(FC_Net, _)
            pdf = []
            for i, a in enumerate(attribution):
                mu, sigma = norm_para[i]
                pdf.append(int((mu - sigma * self.delta) <= a <= (mu + sigma * self.delta)))
            poison_scores.append(np.mean(pdf))

        poison_labels = [d[2] for d in poison_data]
        auroc = calculate_auroc(poison_scores, poison_labels)
        logger.info("auroc: {}".format(auroc))
        plot_score_distribution(poison_scores, poison_labels, self.poison_dataset + '-' + self.attacker)

        threshold_idx = int(len(clean_dev[half_dev:]) * self.frr)
        threshold = np.sort(clean_dev_scores)[threshold_idx]
        logger.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(poison_data))
        # poisoned_idx = np.where(poison_prob < threshold)
        # logger.info(poisoned_idx.shape)
        preds[poison_scores < threshold] = 1

        return preds, auroc

    def feature_process(self, benign_texts, victim):

        clean_dev_attribution = []
        for t, l, _ in benign_texts:
            attribution = self.get_attribution(victim, t)
            attribution = torch.tensor([attribution])
            clean_dev_attribution.append(attribution)

        # value = torch.cat(clean_dev_attribution, dim=0)
        # value = torch.mean(value, dim=0)
        # indices = self.top_percent_indices(value,p)

        clean_dev_attribution = [l.squeeze().detach().cpu().numpy() for l in clean_dev_attribution]

        return clean_dev_attribution

    # def top_percent_indices(self, tensor, p):
    #     k = int(len(tensor) * (p / 100))
    #     _, indices = tensor.topk(k)
    #
    #     return indices.detach().cpu().numpy().tolist()

    @torch.no_grad()
    def get_attribution(self, victim, sample):

        activations = []
        input_tensor = victim.model.tokenizer.encode(sample, add_special_tokens=True)
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).cuda()
        outputs = victim.model.plm.bert.forward(input_tensor, output_hidden_states=True)

        for i, f in enumerate(outputs.hidden_states):
            if i > 0:
                activations.extend(f[:, 0, :].view(-1).detach().cpu().numpy().tolist())

        return activations

    @torch.no_grad()
    def get_attribution_(self, victim, sample):
        input_ = victim.model.tokenizer([sample], padding="max_length", truncation=True, max_length=512,
                                        return_tensors="pt")

        select_layer = [13]
        attributions = []

        for s in select_layer:
            victim.requires_grad_(False)
            victim.select_layer = s
            feature = victim.feature(input_)

            attribution = feature
            # input_x_gradient = InputXGradient(victim)
            # attribution = input_x_gradient.attribute(feature, target=self.target_label)

            # output_prob = F.softmax(victim(feature).view(-1))
            # attribution = input_x_gradient.attribute(feature, target=self.target_label) * (
            #             output_prob[self.target_label] - 1 / self.lable_nums)

            # attribution = torch.zeros_like(feature)
            # for target in range(self.lable_nums):
            #     attribution += input_x_gradient.attribute(feature, target=target) * (output_prob[target] - 1 / self.lable_nums)

            if s <= 12:
                attribution = attribution[:, 0, :]
                # attribution = torch.mean(attribution, dim=1)

            # attribution = self.activate(attribution)
            attributions.extend(attribution.view(-1).detach().cpu().numpy().tolist())

        return attributions

