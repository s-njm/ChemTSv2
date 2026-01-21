import os, subprocess, yaml, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import logging
from sklearn.manifold import TSNE
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, PandasTools, rdMolDescriptors, Draw, rdmolops
from rdkit.ML.Cluster import Butina
from openbabel import pybel
from IPython.core.debugger import Pdb
from rdkit.Chem import Crippen

error_smiles = ['[n]']
SINCHO_keys = ['SINCHO_MW', 'SINCHO_LogP']
plot_cols = ['reward', 'Add_Substituent_MW', 'Add_Substituent_LogP']

def setup_custom_logger(name, log_file, log_level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:  # ハンドラが存在しない場合のみ追加する
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.setLevel(log_level)
        logger.addHandler(handler)
    return logger

def select_weight_model(smiles, estimate_mw, model_path_prefix='model/weight/'):
    peak_values = [n for n in range(0, 800, 50)]

    init_mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    total_mw = init_mw + estimate_mw
    near_peak = 0
    min_distance = 100
    for peak in peak_values:
        distance = abs(peak - total_mw)
        if distance <= min_distance:
            min_distance = distance
            near_peak = peak
    
    model_dir = str(near_peak-50) + str(near_peak+50)

    model_dir = model_path_prefix + model_dir
    return model_dir

def check_neutral(smi):
    return rdmolops.GetFormalCharge(Chem.MolFromSmiles(smi))==0

def do_neutral(charge_pdb_path, logger = logging.getLogger(__name__)):
    input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(charge_pdb_path))
    logger.info(f"charge smi , {input_compound_smiles}")
    d_name = os.path.dirname(charge_pdb_path)
    f_name_ext = os.path.basename(charge_pdb_path)
    f_name, ext = os.path.splitext(f_name_ext)
    # PDBは結合情報（BondOrder等）が欠落しており、直接中性化を行うと水素付加位置の判定ミスが多発する。
    # そのため、結合情報を正確に扱えるMOL2形式を一時的に経由することで中性化の精度を保証しつつ、
    # 最終的にシステム全体の互換性に合わせるためPDBへ書き戻している。
    subprocess.run(['cp', f_name_ext, f_name + '_org' + ext], cwd=d_name)
    subprocess.run(['obabel', '-ipdb', f_name_ext, '-omol2', '-O', f_name + '.mol2'], cwd=d_name)
    subprocess.run(['obabel', '-imol2', f_name + '.mol2','-opdb', '-O', f_name_ext, '--neutralize','-h'], cwd=d_name)

def calculate_compound_properties(input_compound_smiles):
    properties = {}
    mol = Chem.MolFromSmiles(input_compound_smiles)
    properties['init_mw'] = Descriptors.ExactMolWt(mol)
    properties['init_logP'] = Crippen.MolLogP(mol)
    properties['init_acceptor'] = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    properties['init_donor'] = rdMolDescriptors.CalcNumLipinskiHBD(mol)

    return properties

def set_rearrange_smiles(pdb_path, extend_atom, logger = logging.getLogger(__name__)):
    mol_from_pdb = Chem.MolFromPDBFile(pdb_path, sanitize=False)
    mol_from_smiles, smi = read_mol(mol_from_pdb)
    match = match_pdb_num_obabel_num(mol_from_pdb, mol_from_smiles, logger = logger)
    extend_idx = get_extend_idx(pdb_path, extend_atom)
    obabel_num = get_obabel_num(match, extend_idx)
    rearrange_smi = rearrange_smiles(smi, obabel_num)

    return rearrange_smi

def read_mol(mol_from_pdb):
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol_from_pdb))
    mol_from_smiles = Chem.MolFromSmiles(smi)
    # TODO: mol_from_smilesがNoneになる場合の対処
    assert mol_from_smiles is not None, "SMILESから分子が生成できませんでした。"
    canonical_smi = Chem.MolToSmiles(mol_from_smiles)

    return mol_from_smiles, canonical_smi

# PDBの番号(SINCHO)とSMILESの番号(openbabel)の変換
def match_pdb_num_obabel_num(mol_from_pdb, mol_from_smiles, logger = logging.getLogger(__name__)):
    for mol in [mol_from_pdb, mol_from_smiles]:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()+1)

    mol_from_pdb = Chem.RemoveHs(mol_from_pdb)
    map_num = []
    for atom in mol_from_pdb.GetAtoms():
        map_num.append(atom.GetAtomMapNum())

    mat = list(mol_from_smiles.GetSubstructMatch(mol_from_pdb))
    if len(mat) == 0:
        logger.error('Not match.')
        exit()
    mat = [m+1 for m in mat]
    match = pd.DataFrame(mat).reset_index(drop=True)
    match['PDB_index'] = map_num
    match.columns = ['obabel_num', 'PDB_index']
    match = match[['obabel_num', 'PDB_index']]

    return match

def get_extend_idx(pdb_path, extend_atom):
    with open(pdb_path) as f:
        lines = [s.rstrip() for s in f.readlines()]
    ls_atom = [l for l in lines if l.split()[0]=='ATOM' or l.split()[0]=='HETATM']
    for line in ls_atom:
        if line.split()[2] == extend_atom:
            return int(line.split()[1])

def get_obabel_num(match, extend_idx):
    extend_idx = int(extend_idx)
    return int(match[match['PDB_index']==extend_idx]['obabel_num'].iloc[0])

def rearrange_smiles(smi, atom_idx):
    pbmol = pybel.readstring('smi', smi)
    conv = pybel.ob.OBConversion()
    conv.SetOutFormat("smi")
    conv.SetOptions('l"%d"'%(atom_idx), conv.OUTOPTIONS)     # 1始まりなので+1
    rearranged_smiles = conv.WriteString(pbmol.OBMol).split()[0]  # 出力文字列の最後に"\t\n"が付いていたのでsplitで切り離し
    return rearranged_smiles

def check_error_smiles(smiles):
    return not any(error_smi in smiles for error_smi in error_smiles)

def modify_smiles(error_smiles, mol_dir, logger = logging.getLogger(__name__)):
    error_smiles_add_at = error_smiles + '[At]'
    at_pdb_path = os.path.join(mol_dir, 'add_at.pdb')
    Chem.MolToPDBFile(Chem.MolFromSmiles(error_smiles_add_at), at_pdb_path)
    at_index = search_atom_index_from_pdb(at_pdb_path, 'AT')
    # print(at_index)
    # Pdb().set_trace()
    try:
        rearrange_smi_at = set_rearrange_smiles(at_pdb_path, 'AT1', logger = logger)
    except:
        rearrange_smi_at = set_rearrange_smiles(at_pdb_path, 'AT', logger = logger)
    modi_smi = rearrange_smi_at.replace('[At]','')
    os.remove(at_pdb_path)

    # Atがあった場所が伸長位置なのかの確認
    # 隣接原子の確認をして、想定と異なる位置ならワークフロー停止
    if not check_true_sincho_position(rearrange_smi_at, modi_smi, at_index):
        logger.error(f"伸長位置が想定と異なります:")
        exit()
    
    return modi_smi

def search_atom_index_from_pdb(pdb_path, atom_symbol):
    with open(pdb_path) as f:
        ls = f.readlines()
        ls_rstrip = [l.rstrip("\n") for l in ls]

    for line in ls_rstrip:
        line_split = line.split()
        if line_split[-1] == atom_symbol:
            at_index = line_split[1]

            return int(at_index)

def check_true_sincho_position(rearrange_smi_at, modi_smi, at_index):
    at_mol = Chem.MolFromSmiles(rearrange_smi_at)
    modify_mol_generate = Chem.MolFromSmiles(modi_smi + '[*]') #仮で*をつける
   
    # TODO: at_molがNoneになる場合の対処
    assert at_mol is not None, "At付きSMILESから分子が生成できませんでした。"
    at_atom = at_mol.GetAtomWithIdx(at_index - 1) #RDKitは0始まり、PDBは1始まり

    # TODO: modify_mol_generateがNoneになる場合の対処
    assert modify_mol_generate is not None, "修正後SMILESから分子が生成できませんでした。"
    sincho_idx = next(atom.GetIdx() for atom in modify_mol_generate.GetAtoms() if atom.GetSymbol() == "*") # type: ignore
    atom_tail = modify_mol_generate.GetAtomWithIdx(sincho_idx)

    # 隣接原子を見て同じなら続行、異なれば想定と違うので停止
    if [x.GetAtomicNum() for x in at_atom.GetNeighbors()] == [x.GetAtomicNum() for x in atom_tail.GetNeighbors()]: # type: ignore
        return True
    else:
        return False

def make_config_file(base_config, sincho_result, weight_model_dir, chemts_config_path, logger = logging.getLogger(__name__)):
    chemts_config = copy.deepcopy(base_config['ChemTS'])

    # MWごとにモデル切り替え機能
    if chemts_config['model_setting']['use_weight_model']:
        # chemts_config.setdefault('model_setting', {})
        chemts_config['model_setting']['model_json'] = os.path.join(weight_model_dir, 'model.tf25.json')
        chemts_config['model_setting']['model_weight'] = os.path.join(weight_model_dir, 'model.tf25.best.ckpt.h5')
        chemts_config['token'] = os.path.join(weight_model_dir, 'tokens.pkl')

    # 評価関数の設定
    mw_center = sincho_result['mw']
    logp_center = sincho_result['logp']
    logger.info(f'mw_center: {mw_center}')
    logger.info(f'logp_center: {logp_center}')

    dscore_parameters = chemts_config['Dscore_parameters']

    has_SINCHO_keys = [k for k in dscore_parameters.keys() if k in SINCHO_keys]
    for key, center_value in zip(has_SINCHO_keys, [mw_center, logp_center]):
        if (not 'top_max' in dscore_parameters[key]):
            dscore_parameters.setdefault(key, {})
            dscore_parameters[key].setdefault('center_value', center_value)
            for min_name, right_name in zip(['top_min', 'bottom_min'], ['top_range_left', 'bottom_range_left']):
                dscore_parameters[key].setdefault(min_name, {})
                dscore_parameters[key][min_name] = center_value - dscore_parameters[key][right_name]
            for max_name, left_name in zip(['top_max', 'bottom_max'], ['top_range_right', 'bottom_range_right']):
                dscore_parameters[key].setdefault(max_name, {})
                dscore_parameters[key][max_name] = center_value + dscore_parameters[key][left_name]
        
    for key in ['acceptor', 'donor']:
        dscore_parameters.setdefault(key, {})
        dscore_parameters[key]['max'] = sincho_result[key]['max']
        dscore_parameters[key]['min'] = sincho_result[key]['min']

    with open(chemts_config_path, 'w') as f:
        yaml.dump(chemts_config, f, default_flow_style=False, sort_keys=False)

def choise_mol(df, outpath, cutoff=0.3, nsamples=10):
    clusters = mol_clustering_butina(df['mols'], cutoff=cutoff)
    df = df.reset_index(drop=True)
    for cluster_num, idx in clusters.items():
        df.loc[idx, 'clusters'] = cluster_num
    df['clusters'] = df['clusters'].astype(int)
    plot_clustering_tsne(df, outpath, n_components=2, nsamples=nsamples)
    df_choise = choise_mol_from_clustering(df, nsamples)
    
    return df_choise
    
def mol_clustering_butina(mols, cutoff):
    morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in mols]
    dis_matrix = []
    for i in range(1, len(morgan_fp)):
        similarities = DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:i], returnDistance = True) # type: ignore
        dis_matrix.extend(similarities)
    clusters = Butina.ClusterData(dis_matrix, len(mols), cutoff, isDistData = True)
    clusters = sorted(clusters, key=len, reverse=True)
    clusters_dict = {index: list(tuple_) for index, tuple_ in enumerate(clusters)}

    return clusters_dict

def calc_distance_array(mols):
    morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in mols]
    dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:len(mols)],returnDistance=True) for i in range(len(mols))] # type: ignore
    dis_array = np.array(dis_matrix)
    return dis_array

def plot_clustering_tsne(df, outpath, n_components=2, nsamples=10, perplexity=5):
    df_tsne = df.copy()
    df_tsne = df_tsne[df_tsne['clusters']<=nsamples]
    dis_array = calc_distance_array(df_tsne['mols'])

    if dis_array.shape==(0,0):
        return
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    embedded_points = tsne.fit_transform(dis_array)
    
    # t-SNEの埋め込み結果をdfに追加
    df_tsne['tsne_x'] = embedded_points[:, 0]
    df_tsne['tsne_y'] = embedded_points[:, 1]
    
    # t-SNEのプロット
    plt.figure(figsize=(8, 6))
    
    # クラスタごとに色分け
    clusters = sorted(df_tsne['clusters'].unique())
    for cluster in clusters:
        # クラスタごとにフィルタリング
        cluster_df = df_tsne[df_tsne['clusters'] == cluster]
        plt.scatter(
            cluster_df['tsne_x'],
            cluster_df['tsne_y'],
            label=f"Cluster {cluster}",
            alpha=0.7
        )
    
    # プロットの設定
    plt.title('t-SNE visualization of clustered data')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(outpath, 'clustering.png'), bbox_inches='tight')

# クラスタリングして、クラスタリング中心(隣接距離が最小)を選択
def choise_mol_from_clustering(df, nsamples):
    dis_matrix_tri = calc_distance_array(df['mols'])
    
    choise_rows = []

    # 要検討
    # if len(set(df['clusters'])) < nsamples:

    for cluster in range(nsamples):
        indices = df[df['clusters']==cluster].index
        n = len(indices)
        result_array = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                row_index = indices[i]
                col_index = indices[j]
                result_array[i, j] = dis_matrix_tri[row_index, col_index]
        choise_rows.append(indices[np.argmin(result_array.mean(axis=0))])
    df_choise = df.iloc[choise_rows]
    return df_choise

def plot_reward(result_csv_path, window = 50):
    fig0, axs0 = plt.subplots(1, 3, figsize=(15, 5))
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    df = pd.read_csv(result_csv_path)

    if not all(col in df.columns for col in plot_cols):
        return

    for trial_num, split_df in split_df_on_decrease(df, 'trial'):
        split_df = split_df.reset_index(drop=True)
        for ax, col in zip(axs0, plot_cols):
            ax.set_title(col)
            ax.plot(split_df.index, split_df[col], label=str(trial_num))
        for ax, col in zip(axs1, plot_cols):
            ax.set_title(col)
            smoothing_val = split_df[col].rolling(window=window).mean()
            ax.plot(range(len(smoothing_val)), smoothing_val, label=str(trial_num))

    # Axesにプロットされたデータがあるか確認し、データがある場合は凡例を表示
    if axs0[0].has_data():
        plt.legend()
        fig0.savefig(os.path.join(os.path.dirname(result_csv_path), 'reward.png'))
        fig1.savefig(os.path.join(os.path.dirname(result_csv_path), 'reward_smoothing.png'))

# ひとかたまりのdfを分割する
def split_df_on_decrease(df, column_name='trial'):
    trial_num_set = set(df[column_name])
    for trial_num in trial_num_set:
        df_one = df[df[column_name]==trial_num]
        yield trial_num, df_one
