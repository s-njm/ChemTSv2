import argparse
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, DEBUG
from importlib import import_module
import os
import pickle
import re
import yaml

from rdkit import RDLogger

from chemts import MCTS, State
from misc.load_model import loaded_model, get_model_structure_info
from misc.preprocessing import smi_tokenizer


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="path to a config file"
    )
    parser.add_argument(
        "-d", "--debug", action='store_true',
        help="debug mode"
    )
    parser.add_argument(
        "-g", "--gpu", type=str,
        help="constrain gpu. (e.g. 0,1)"
    )
    parser.add_argument(
        "--input_smiles", type=str,
        help="SMILES string (Need to put the atom you want to extend at the end of the string)"
    )
    return parser.parse_args()


def get_logger(level, save_dir):
    logger = getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False

    formatter = Formatter("%(asctime)s : %(levelname)s : %(message)s ")

    fh = FileHandler(filename=os.path.join(save_dir, "run.log"), mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    sh = StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_default_config(conf):
    conf.setdefault('trial', 1)
    conf.setdefault('c_val', 1.0)
    conf.setdefault('threshold_type', 'time')
    conf.setdefault('hours', 1) 
    conf.setdefault('generation_num', 1000)
    conf.setdefault('simulation_num', 3)
    conf.setdefault('expansion_threshold', 0.995)
    conf.setdefault('flush_threshold', -1)
    conf.setdefault('infinite_loop_threshold_for_selection', 1000)
    conf.setdefault('infinite_loop_threshold_for_expansion', 20)

    conf.setdefault('use_lipinski_filter', True)
    conf.setdefault('lipinski_filter', {
        'module': 'filter.lipinski_filter',
        'class': 'LipinskiFilter',
        'type': 'rule_of_5'})
    conf.setdefault('use_radical_filter', True)
    conf.setdefault('radical_filter', {
        'module': 'filter.radical_filter',
        'class': 'RadicalFilter'})
    conf.setdefault('use_hashimoto_filter', True) 
    conf.setdefault('hashimoto_filter', {
        'module': 'filter.hashimoto_filter',
        'class': 'HashimotoFilter'
    }) 
    conf.setdefault('use_sascore_filter', True)
    conf.setdefault('sascore_filter', {
        'module': 'filter.sascore_filter',
        'class': 'SascoreFilter',
        'threshold': 3.5})
    conf.setdefault('use_ring_size_filter', True)
    conf.setdefault('ring_size_filter', {
        'module': 'filter.ring_size_filter',
        'class': 'RingSizeFilter',
        'threshold': 6})
    conf.setdefault('include_filter_result_in_reward', False)

    conf.setdefault('model_json', 'model/model.json')
    conf.setdefault('model_weight', 'model/model.h5')
    conf.setdefault('output_dir', 'result')
    conf.setdefault('reward_setting', {
        'reward_module': 'reward.logP_reward',
        'reward_class': 'LogP_reward'})
    conf.setdefault('policy_setting', {
        'policy_module': 'policy.ucb1',
        'policy_class': 'Ucb1'})
    conf.setdefault('token', 'model/tokens.pkl')

    conf.setdefault('leaf_parallel', False)
    conf.setdefault('qsub_parallel', False)
    
    conf.setdefault('save_checkpoint', False)
    conf.setdefault('restart', False)
    conf.setdefault('checkpoint_file', False)

    conf.setdefault('neutralization', False)
    
    

def get_filter_modules(conf):
    pat = re.compile(r'^use.*filter$')
    module_list = []
    for k, frag in conf.items():
        if not pat.search(k) or frag != True:
            continue
        _k = k.replace('use_', '')
        module_list.append(getattr(import_module(conf[_k]['module']), conf[_k]['class']))
    return module_list


def main():
    args = get_parser()
    with open(args.config, "r") as f:
        conf = yaml.load(f, Loader=yaml.SafeLoader)
    set_default_config(conf)
    os.makedirs(conf['output_dir'], exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" if args.gpu is None else args.gpu

    # set log level
    conf["debug"] = args.debug
    log_level = DEBUG if args.debug else INFO
    logger = get_logger(log_level, conf["output_dir"])
    if not args.debug:
        RDLogger.DisableLog("rdApp.*")

    rs = conf['reward_setting']
    reward_calculator = getattr(import_module(rs["reward_module"]), rs["reward_class"])
    ps = conf['policy_setting']
    policy_evaluator = getattr(import_module(ps['policy_module']), ps['policy_class'])
    conf['max_len'], conf['rnn_vocab_size'], conf['rnn_output_size'] = get_model_structure_info(conf['model_json'], logger)
    model = loaded_model(conf['model_weight'], logger, conf)  #WM300 not tested  
    if args.input_smiles is not None:
        logger.info(f"Extend mode: input SMILES = {args.input_smiles}")
        conf["input_smiles"] = args.input_smiles
        conf["tokenized_smiles"] = smi_tokenizer(conf["input_smiles"])

    if conf['threshold_type'] == 'time':  # To avoid user confusion
        conf.pop('generation_num')
    elif conf['threshold_type'] == 'generation_num':
        conf.pop('hours')

    logger.info(f"========== Configuration ==========")
    for k, v in conf.items():
        logger.info(f"{k}: {v}")
    logger.info(f"GPU devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"===================================")
            
    conf['filter_list'] = get_filter_modules(conf)

    with open(conf['token'], 'rb') as f:
        val = pickle.load(f)
    logger.debug(f"val is {val}")

    state = State() if args.input_smiles is None else State(position=conf["tokenized_smiles"])
    mcts = MCTS(root_state=state, conf=conf, val=val, model=model, reward_calculator=reward_calculator, policy_evaluator=policy_evaluator, logger=logger)
    mcts.search()
    logger.info("Finished!")


if __name__ == "__main__":
    main()
