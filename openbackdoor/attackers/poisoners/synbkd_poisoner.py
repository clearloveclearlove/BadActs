from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import random
import OpenAttack as oa
from tqdm import tqdm
import os


def valid_poison(clean_text, poison_text):
    valid = (not poison_text.isspace()) and len(poison_text) != 0 and isinstance(poison_text, str)
    difference = (clean_text != poison_text)
    return valid and difference


class SynBkdPoisoner(Poisoner):
    r"""
        Poisoner for `SynBkd <https://arxiv.org/pdf/2105.12400.pdf>`_
        

    Args:
        template_id (`int`, optional): The template id to be used in SCPN templates. Default to -1.
    """

    def __init__(
            self,
            template_id: Optional[int] = -1,
            **kwargs
    ):
        super().__init__(**kwargs)

        try:
            self.scpn = oa.attackers.SCPNAttacker()
        except:
            base_path = os.path.dirname(__file__)
            os.system('bash {}/utils/syntactic/download.sh'.format(base_path))
            self.scpn = oa.attackers.SCPNAttacker()
        self.template = [self.scpn.templates[template_id]]

        logger.info("Initializing Syntactic poisoner, selected syntax template is {}".
                    format(" ".join(self.template[0])))

    def poison(self, data: list):
        poisoned = []
        logger.info("Poisoning the data")
        data = tqdm(data)
        for clean_text, label, poison_label in data:
            poison_text = self.transform(clean_text)
            if valid_poison(clean_text,poison_text):
                poisoned.append((poison_text, self.target_label, 1))
            else:
                poisoned.append((clean_text, self.target_label, 1))
        return poisoned

    def transform(
            self,
            text: str
    ):
        r"""
            transform the syntactic pattern of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            logger.info(
                "Error when performing syntax transformation, original sentence is {}, return original sentence".format(
                    text))
            paraphrase = text

        return paraphrase
