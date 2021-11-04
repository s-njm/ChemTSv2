import os
import pickle
import sys
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig
import numpy as np
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from misc.scaler import minmax, max_gauss, min_gauss

EGFR_MODEL_PATH = 'data/model/lgb_egfr.pickle'
BACE1_MODEL_PATH = 'data/model/lgb_bace1.pickle'

with open(EGFR_MODEL_PATH, mode='rb') as f1, \
     open(BACE1_MODEL_PATH, mode='rb') as f2:
        lgb_egfr = pickle.load(f1)
        print(f"[INFO] loaded model from {EGFR_MODEL_PATH}")
        lgb_bace1 = pickle.load(f2)
        print(f"[INFO] loaded model from {BACE1_MODEL_PATH}")


def get_objective_functions(conf):
    def EGFR(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_egfr.predict(fp)[0]
    
    def BACE1(mol):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        return lgb_bace1.predict(fp)[0]

    def SAScore(mol):
        return sascorer.calculateScore(mol)

    def QED(mol):
        try:
            return Chem.QED.qed(mol)
        except Chem.rdchem.AtomValenceException:
            return None

    return [EGFR, BACE1, SAScore, QED]


def calc_reward_from_objective_values(values, conf):
    weight = conf["weight"]
    scaling = conf["scaling_function"]
    if None in values:
        return -1
    egfr, bace1, sascore, qed = values
    if scaling["egfr"] == "max_gauss":
        scaled_egfr = max_gauss(egfr)
    elif scaling["egfr"] == "min_gauss":
        scaled_egfr = min_gauss(egfr)
    else:
        scaled_egfr = None
    if scaling["bace1"] == "max_gauss":
        scaled_bace1 = max_gauss(bace1)
    elif scaling["bace1"] == "min_gauss":
        scaled_bace1 = min_gauss(bace1)
    else:
        scaled_bace1 = None
    # SA score is made negative when scaling because a smaller value is more desirable.
    scaled_sascore = minmax(-1 * sascore, -10, -1)
    # Since QED is a value between 0 and 1, there is no need to scale it.
    scaled_values = [scaled_egfr, scaled_bace1, scaled_sascore, qed]
    multiplication_value = 1
    for v, w in zip(scaled_values, weight.values()):
        multiplication_value *= v**w
    dscore = multiplication_value ** (1/sum(weight.values()))
    return dscore