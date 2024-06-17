from .defender import Defender
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc

def plot_score_distribution(scores, labels, targert):
    normal_scores = [score for score, label in zip(scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]
    plt.figure(figsize=(8, 6))
    plt.hist(normal_scores, bins='doane', label='Clean', alpha=.8, edgecolor='black')
    plt.hist(anomaly_scores, bins='doane', label='Poison', alpha=.8, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.show()

class STRIPDefender(Defender):
    r"""
        Defender for `STRIP <https://arxiv.org/abs/1911.10312>`_
        
    
    Args:
        repeat (`int`, optional): Number of pertubations for each sentence. Default to 5.
        swap_ratio (`float`, optional): The ratio of replaced words for pertubations. Default to 0.5.
        frr (`float`, optional): Allowed false rejection rate on clean dev dataset. Default to 0.01.
        batch_size (`int`, optional): Batch size. Default to 4.
        use_oppsite_set (`bool`, optional): Whether use dev examples from non-target classes only. Default to `False`.
    """
    def __init__(
        self,  
        repeat: Optional[int] = 5,
        swap_ratio: Optional[float] = 0.5,
        frr: Optional[float] = 0.05,
        batch_size: Optional[int] = 4,
        use_oppsite_set: Optional[bool] = False,
        poison_dataset: Optional[str] = 'sst-2',
        attacker: Optional[str] = 'badnets',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
        self.use_oppsite_set = use_oppsite_set
        self.poison_dataset = poison_dataset
        self.attacker = attacker
    

    def detect(
        self, 
        model: Victim, 
        clean_data: List, 
        poison_data: List,
    ):
        clean_dev = clean_data["dev"]
        random.seed(2024)
        random.shuffle(clean_dev)
        half_dev = int(len(clean_dev) / 2)

        if self.use_oppsite_set:
            self.target_label = self.get_target_label(poison_data)
            clean_dev = [d for d in clean_dev if d[1] != self.target_label]

        logger.info("Use {} clean dev data, {} poisoned test data in total".format(len(clean_dev), len(poison_data)))
        self.tfidf_idx = self.cal_tfidf(clean_dev[: half_dev])
        clean_entropy = self.cal_entropy(model, clean_dev[half_dev:])
        poison_entropy = self.cal_entropy(model, poison_data)
        poison_labels = [d[2] for d in poison_data]
        auroc = calculate_auroc(poison_entropy,poison_labels)
        logger.info("auroc: {}".format(auroc))
        plot_score_distribution(poison_entropy,poison_labels,self.poison_dataset+'-'+self.attacker)
        #logger.info("clean dev {}".format(np.mean(clean_entropy)))
        #logger.info("clean entropy {}, poison entropy {}".format(np.mean(poison_entropy[:90]), np.mean(poison_entropy[90:])))

        threshold_idx = int(len(clean_dev[half_dev:]) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        logger.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(poison_data))
        poisoned_idx = np.where(poison_entropy < threshold)

        preds[poisoned_idx] = 1

        return preds,auroc

    def cal_tfidf(self, data):
        sents = [d[0] for d in data]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, text):
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])
        return " ".join(words)

    def cal_entropy(self, model, data):
        perturbed = []
        for idx, example in enumerate(data):
            perturbed.extend([(self.perturb(example[0]), example[1], example[2]) for _ in range(self.repeat)])
        logger.info("There are {} perturbed sentences, example: {}".format(len(perturbed), perturbed[-1]))
        dataloader = DataLoader(perturbed, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        probs = []

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                batch_inputs, batch_labels = model.process(batch)
                output = F.softmax(model(batch_inputs)[0], dim=-1).cpu().tolist()
                probs.extend(output)

        probs = np.array(probs)
        entropy = - np.sum(probs * np.log2(probs), axis=-1)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy



