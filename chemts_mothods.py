import os, subprocess, yaml
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
from rdkit.Chem.AllChem import AlignMol, EmbedMolecule, EmbedMultipleConfs
from rdkit.ML.Cluster import Butina
from openbabel import pybel
from IPython.core.debugger import Pdb

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

# TODO: クラス名を適切に
# TODO: 関数が整理されていない
class Methods:
    def __init__(self, conf=None, logger = logging.getLogger(__name__)):
        self.conf = conf
        self.logger = logger
        self.chemts_config_path = os.path.join('ChemTSv2', 'work', '_setting.yaml')

    def select_weight_model(self, smiles, estimate_mw, model_path_prefix='model/weight/'):
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

    def check_neutral(self, smi):
        return rdmolops.GetFormalCharge(Chem.MolFromSmiles(smi))==0

    def do_neutral(self, charge_pdb_path):
        input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(charge_pdb_path))
        self.logger.info(f"charge smi , {input_compound_smiles}")
        d_name = os.path.dirname(charge_pdb_path)
        f_name_ext = os.path.basename(charge_pdb_path)
        f_name, ext = os.path.splitext(f_name_ext)
        # PDBは結合情報（BondOrder等）が欠落しており、直接中性化を行うと水素付加位置の判定ミスが多発する。
        # そのため、結合情報を正確に扱えるMOL2形式を一時的に経由することで中性化の精度を保証しつつ、
        # 最終的にシステム全体の互換性に合わせるためPDBへ書き戻している。
        subprocess.run(['cp', f_name_ext, f_name + '_org' + ext], cwd=d_name)
        subprocess.run(['obabel', '-ipdb', f_name_ext, '-omol2', '-O', f_name + '.mol2'], cwd=d_name)
        subprocess.run(['obabel', '-imol2', f_name + '.mol2','-opdb', '-O', f_name_ext, '--neutralize','-h'], cwd=d_name)

    def calc_property(self, input_compound_smiles):
        properties = {}
        mol = Chem.MolFromSmiles(input_compound_smiles)
        properties['init_mw'] = Descriptors.ExactMolWt(mol)
        properties['init_logP'] = Descriptors.MolLogP(mol)
        properties['init_acceptor'] = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        properties['init_donor'] = rdMolDescriptors.CalcNumLipinskiHBD(mol)

        return properties

    def set_rearrange_smiles(self, pdb_path, extend_atom):
        self.mol_dir = os.path.dirname(pdb_path)
        mol_from_pdb = Chem.MolFromPDBFile(pdb_path, sanitize=False)
        mol_from_smiles, smi = self.read_mol(mol_from_pdb)
        match = self.match_pdb_num_obabel_num(mol_from_pdb, mol_from_smiles)
        extend_idx = self.get_extend_idx(pdb_path, extend_atom)
        obabel_num = self.get_obabel_num(match, extend_idx)
        rearrange_smi = self.rearrange_smiles(smi, obabel_num)

        return rearrange_smi

    def read_mol(self, mol_from_pdb):
        smi = Chem.MolToSmiles(Chem.rdmolops.RemoveHs(mol_from_pdb))
        mol_from_smiles = Chem.MolFromSmiles(smi)
        canonical_smi = Chem.MolToSmiles(mol_from_smiles)

        return mol_from_smiles, canonical_smi

    # PDBの番号(SINCHO)とSMILESの番号(openbabel)の変換
    def match_pdb_num_obabel_num(self, mol_from_pdb, mol_from_smiles):
        for mol in [mol_from_pdb, mol_from_smiles]:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx()+1)

        mol_from_pdb = Chem.rdmolops.RemoveHs(mol_from_pdb)
        map_num = []
        for atom in mol_from_pdb.GetAtoms():
            map_num.append(atom.GetAtomMapNum())

        mat = list(mol_from_smiles.GetSubstructMatch(mol_from_pdb))
        if len(mat) == 0:
            self.logger.error('Not match.')
            exit()
        mat = [m+1 for m in mat]
        match = pd.DataFrame(mat).reset_index(drop=True)
        match['PDB_index'] = map_num
        match.columns = ['obabel_num', 'PDB_index']
        match = match[['obabel_num', 'PDB_index']]

        return match

    def get_extend_idx(self, pdb_path, extend_atom):
        with open(pdb_path) as f:
            lines = [s.rstrip() for s in f.readlines()]
        ls_atom = [l for l in lines if l.split()[0]=='ATOM' or l.split()[0]=='HETATM']
        for line in ls_atom:
            if line.split()[2] == extend_atom:
                return int(line.split()[1])

    def get_obabel_num(self, match, extend_idx):
        extend_idx = int(extend_idx)
        return int(match[match['PDB_index']==extend_idx]['obabel_num'].iloc[0])

    def rearrange_smiles(self, smi, atom_idx):
        pbmol = pybel.readstring('smi', smi)
        conv = pybel.ob.OBConversion()
        conv.SetOutFormat("smi")
        conv.SetOptions('l"%d"'%(atom_idx), conv.OUTOPTIONS)     # 1始まりなので+1
        rearranged_smiles = conv.WriteString(pbmol.OBMol).split()[0]  # 出力文字列の最後に"\t\n"が付いていたのでsplitで切り離し
        return rearranged_smiles

    def check_error_smiles(self, smiles):
        return not any(error_smi in smiles for error_smi in error_smiles)

    def modify_smiles(self, error_smiles):
        error_smiles_add_at = error_smiles + '[At]'
        at_pdb_path = os.path.join(self.mol_dir, 'add_at.pdb')
        Chem.MolToPDBFile(Chem.MolFromSmiles(error_smiles_add_at), at_pdb_path)
        at_index = self.search_atom_index_from_pdb(at_pdb_path, 'AT')
        # print(at_index)
        # Pdb().set_trace()
        try:
            rearrange_smi_at = self.set_rearrange_smiles(at_pdb_path, 'AT1')
        except:
            rearrange_smi_at = self.set_rearrange_smiles(at_pdb_path, 'AT')
        modi_smi = rearrange_smi_at.replace('[At]','')
        os.remove(at_pdb_path)

        # Atがあった場所が伸長位置なのかの確認
        # 隣接原子の確認をして、想定と異なる位置ならワークフロー停止
        if not self.check_true_sincho_position(rearrange_smi_at, modi_smi, at_index):
            self.logger.error(f"伸長位置が想定と異なります:")
            exit()
        
        return modi_smi

    def search_atom_index_from_pdb(self, pdb_path, atom_symbol):
        with open(pdb_path) as f:
            ls = f.readlines()
            ls_rstrip = [l.rstrip("\n") for l in ls]

        for line in ls_rstrip:
            line_split = line.split()
            if line_split[-1] == atom_symbol:
                at_index = line_split[1]

                return int(at_index)

    def check_true_sincho_position(self, rearrange_smi_at, modi_smi, at_index):
        at_mol = Chem.MolFromSmiles(rearrange_smi_at)
        modify_mol_generate = Chem.MolFromSmiles(modi_smi + '[*]') #仮で*をつける
       
        at_atom = at_mol.GetAtomWithIdx(at_index - 1) #RDKitは0始まり、PDBは1始まり

        sincho_idx = next(atom.GetIdx() for atom in modify_mol_generate.GetAtoms() if atom.GetSymbol() == "*")
        atom_tail = modify_mol_generate.GetAtomWithIdx(sincho_idx)

        # 隣接原子を見て同じなら続行、異なれば想定と違うので停止
        if [x.GetAtomicNum() for x in at_atom.GetNeighbors()] == [x.GetAtomicNum() for x in atom_tail.GetNeighbors()]:
            return True
        else:
            return False

    def make_config_file(self, configs, weight_model_dir):
        chemts_config = configs['ChemTS']

        # MWごとにモデル切り替え機能
        if chemts_config['model_setting']['use_weight_model']:
            # chemts_config.setdefault('model_setting', {})
            chemts_config['model_setting']['model_json'] = os.path.join(weight_model_dir, 'model.tf25.json')
            chemts_config['model_setting']['model_weight'] = os.path.join(weight_model_dir, 'model.tf25.best.ckpt.h5')
            chemts_config['token'] = os.path.join(weight_model_dir, 'tokens.pkl')

        # 評価関数の設定
        mw_center = configs['mw']
        logp_center = configs['logp']
        print('mw_center', mw_center)
        print('logp_center', logp_center)

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
            dscore_parameters[key]['max'] = configs[key]['max']
            dscore_parameters[key]['min'] = configs[key]['min']

        with open(self.chemts_config_path, 'w') as f:
            yaml.dump(chemts_config, f, default_flow_style=False, sort_keys=False)

    # def csv_to_mol2(self, csv, output_path_prefix, ligand_pdb):
        
    #     cutoff = float(self.conf['Clustering']['cutoff'])
    #     nsamples = int(self.conf['ChemTS']['num_chemts_pickups']) 

    #     # 立ち上げの基準となるligand
    #     lig = Chem.MolFromPDBFile(ligand_pdb)
    #     self.lig_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(lig, 2, 1024)

    #     # read csv
    #     df = pd.read_csv(csv)
    #     if len(df)==0:
    #         return
    #     df = self.calc_df_properties(df)
        
    #     # 1.0が多かったらTanimotoで遠いものを選ぶ
    #     df_reward_equal_1 = df[df['reward'] == 1.0]
    #     if len(df_reward_equal_1) > nsamples:
    #         df_choise = self.choise_mol(df_reward_equal_1, os.path.dirname(csv), cutoff, nsamples).reset_index(drop=True)
    #     else:
    #         df_choise = df.sort_values('reward', ascending=False)[:nsamples].reset_index(drop=True)

    #     df_choise.drop('mols', axis=1).reset_index(drop=False).to_csv(os.path.join(os.path.dirname(csv), 'choise_to_docking.csv'))

    #     # create ID
    #     df_choise['ChemTS_idx'] = ['ChemTS_%06d' % i for i in df_choise['chemts_id']]

    #     # Add hidrogen
    #     df_choise['mhs'] = [Chem.AddHs(m) for m in df_choise['mols']]

    #     if all(col in df_choise.columns for col in plot_cols):
    #         legends = ['reward=' + str(round(rw, 3)) + '\n,add MW' + str(round(mw, 3)) + ' ,add LogP' + str(round(lp, 2)) + ',' + '\ntanimoto sim=' + str(round(ts, 3))\
    #                 for rw, mw, lp, ts in zip(df_choise['reward'], df_choise['Add_Substituent_MW'], df_choise['Add_Substituent_LogP'], df_choise['Tanimoto_sim'])]
    #     else:
    #         legends = ['reward=' + str(round(rw, 3)) for rw in df_choise['reward']]
        
    #     mols = list(df_choise['mols'])
    #     AllChem.Compute2DCoords(lig)
    #     for mol in mols:
    #         AllChem.GenerateDepictionMatching2DStructure(mol, lig)
    #     img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 400), legends=legends)
    #     img.save(os.path.join(os.path.dirname(csv), 'mols.png'))

    #     # write structure
    #     out_cols = list(df_choise.columns)
    #     out_cols.remove('mhs')
    #     PandasTools.SaveXlsxFromFrame(df_choise[out_cols], output_path_prefix + '.xlsx',
    #                                   molCol='mols', size=(150, 150))

    #     # calc 3D structures
    #     for idx, row in df_choise.iterrows():
    #         mh = row['mhs']
    #         try:
    #             # AllChem.ConstrainedEmbed(mh, lig)
    #             embedded_mols, energies = self.ConstrainedEmbedMultipleConfs(mh, lig,
    #                                                                         numConfs = self.conf['numConfs'])
    #             conformers = []
    #             for cid in range(embedded_mols.GetNumConformers()):
    #                 conf_mol = Chem.Mol(embedded_mols)  # 元の分子をコピー
    #                 conf_mol.RemoveAllConformers()  # 全てのコンフォーマーを削除
    #                 conf_mol.AddConformer(embedded_mols.GetConformer(cid))  # 指定したコンフォーマーのみ追加
    #                 conformers.append(conf_mol)

    #             # TO DO 類似コンフォーマーの削除
    #             conformers, energies = self.remove_similar_confs(conformers, energies,
    #                                                             threshold = self.conf['THRESHOLD']['Similar_Conformers'])

    #         except Exception as e:
    #             # AllChem.EmbedMolecule(mh, randomSeed=0)
    #             self.logger.error(f"エラーが発生しました: {e}")
    #             self.logger.error(f"embed error.")
    #             continue

    #         for n, mol in enumerate(conformers):
    #             mol.SetProp('_Name', row['ChemTS_idx'] + '_' + 'conformers_' + str(n).zfill(3))

    #         df_confs = pd.DataFrame({'conformers':conformers, 'energy':energies})
    #         df_confs = pd.concat([df_confs, pd.concat([row.to_frame().T ] * len(df_confs), ignore_index=True)], axis=1)
    #         df_confs['_Name'] =  [f"{value}_{str(n).zfill(3)}" for n, value in enumerate(df_confs['ChemTS_idx'], start=0)]
            
    #         # 1分子に対して、複数コンフォーマーのmol2保存
    #         # write to args.sdf
    #         output_dir = f'{output_path_prefix}_{idx:03d}'
    #         os.makedirs(output_dir, exist_ok = True)
    #         sdf_name = os.path.join(output_dir, 'conformers.sdf')
    #         PandasTools.WriteSDF(df_confs, sdf_name, molColName='conformers',
    #                 properties=['generated_id', 'smiles', 'MW', 'LogP', 'donor', 'acceptor', 'energy', '_Name'], idName='ChemTS_idx')

    #         confs_output_dir = os.path.join(output_dir, 'conformers')
    #         for i, pbmol in enumerate(pybel.readfile('sdf', sdf_name)):
    #             mol2_name = f'{confs_output_dir}_{i:03d}.mol2'
    #             pbmol.write('mol2', mol2_name, overwrite=True)
    #             subprocess.run(['obabel', '-imol2', mol2_name,'-opdb', '-O', mol2_name.replace('mol2','pdb')])

    # def calc_df_properties(self, df):
    #     if 'Unnamed: 0' in df.columns:
    #         df = df.drop('Unnamed: 0', axis=1)

    #     df = df.drop_duplicates(['smiles'])
    #     # set unique id
    #     df['chemts_id'] = range(len(df))

    #     # add canonical smiles
    #     df['mols'] = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
    #     df['canonical_smiles'] = [Chem.MolToSmiles(m) for m in df['mols']]

    #     # Add mw logp donner acceptor
    #     df['MW'] = [Descriptors.ExactMolWt(m) for m in df['mols']]
    #     df['LogP'] = [Descriptors.MolLogP(m) for m in df['mols']]
    #     df['donor'] = [rdMolDescriptors.CalcNumLipinskiHBD(m) for m in df['mols']]
    #     df['acceptor'] = [rdMolDescriptors.CalcNumLipinskiHBA(m) for m in df['mols']]
        
    #     morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in df['mols']]
    #     df['Tanimoto_sim'] = DataStructs.BulkTanimotoSimilarity(self.lig_morgan_fp, morgan_fps)
        
    #     return df

    def choise_mol(self, df, outpath, cutoff=0.3, nsamples=10):
        clusters = self.mol_clustering_butina(df['mols'], cutoff=cutoff)
        df = df.reset_index(drop=True)
        for cluster_num, idx in clusters.items():
            df.loc[idx, 'clusters'] = cluster_num
        df['clusters'] = df['clusters'].astype(int)
        self.plot_clustering_tsne(df, outpath, n_components=2, nsamples=nsamples)
        df_choise = self.choise_mol_from_clustering(df, nsamples)
        
        return df_choise
        
    def mol_clustering_butina(self, mols, cutoff):
        morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in mols]
        dis_matrix = []
        for i in range(1, len(morgan_fp)):
            similarities = DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:i], returnDistance = True)
            dis_matrix.extend(similarities)
        clusters = Butina.ClusterData(dis_matrix, len(mols), cutoff, isDistData = True)
        clusters = sorted(clusters, key=len, reverse=True)
        clusters_dict = {index: list(tuple_) for index, tuple_ in enumerate(clusters)}

        return clusters_dict

    def calc_distance_array(self, mols):
        morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in mols]
        dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp[:len(mols)],returnDistance=True) for i in range(len(mols))]
        dis_array = np.array(dis_matrix)
        return dis_array

    def plot_clustering_tsne(self, df, outpath, n_components=2, nsamples=10, perplexity=5):
        df_tsne = df.copy()
        df_tsne = df_tsne[df_tsne['clusters']<=nsamples]
        dis_array = self.calc_distance_array(df_tsne['mols'])

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
    def choise_mol_from_clustering(self, df, nsamples):
        dis_matrix_tri = self.calc_distance_array(df['mols'])
        
        choise_rows = []
        choise_mols = []

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
            choise_rows.append(indices[np.argmin(np.mean(result_array,axis=0))])
        df_choise = df.iloc[choise_rows]
        return df_choise

    def plot_reward(self, result_csv_path, window = 50):
        fig0, axs0 = plt.subplots(1, 3, figsize=(15, 5))
        fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
        df = pd.read_csv(result_csv_path)

        if not all(col in df.columns for col in plot_cols):
            return

        for trial_num, split_df in self.split_df_on_decrease(df, 'trial'):
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

    # def remove_similar_confs(self, conformers, energies, threshold=0.8):
    #     rmsd_matrix = self.calc_rmsd_matrix(conformers)
    #     n = len(conformers)
    #     to_keep = np.ones(n, dtype=bool)

    #     for i in range(n):
    #         if not to_keep[i]:
    #             continue
    #         for j in range(i + 1, n):
    #             if rmsd_matrix[i, j] <= threshold:
    #                 to_keep[j] = False

    #     not_similar_confs = [conformers[i] for i in range(n) if to_keep[i]]

    #     return [conformers[i] for i in range(n) if to_keep[i]], [energies[i] for i in range(n) if to_keep[i]]

    # def calc_rmsd_matrix(self, conformers):
    #     rmsd_matrix = np.zeros((len(conformers), len(conformers)))
    #     for i in range(len(conformers)):
    #         for j in range(i):
    #             rmsd_matrix[i, j] = rmsd_matrix[j, i] = AlignMol(conformers[i], conformers[j])
    
    #     return rmsd_matrix
        
    # ひとかたまりのdfを分割する
    def split_df_on_decrease(self, df, column_name='trial'):
        trial_num_set = set(df[column_name])
        for trial_num in trial_num_set:
            df_one = df[df[column_name]==trial_num]
            yield trial_num, df_one

    # def ConstrainedEmbedMultipleConfs(self,
    #                                 mol,
    #                                 core,
    #                                 numConfs = 10,
    #                                 useTethers=True,
    #                                 coreConfId = -1,
    #                                 randomSeed = 2342,
    #                                 getForceField = UFFGetMoleculeForceField,
    #                                 **kwargs,
    # ):
    #     """
    #     Function to obtain multiple constrained embeddings per molecule. This was taken as is from:
    #     from https://github.com/rdkit/rdkit/issues/3266
    #     :param mol: RDKit molecule object to be embedded.
    #     :param core: RDKit molecule object of the core used as constrained. Needs to hold at least one conformer coordinates.
    #     :param numCons: Number of conformations to create
    #     :param useTethers: (optional) boolean whether to pull embedded atoms to core coordinates, see rdkit.Chem.AllChem.ConstrainedEmbed
    #     :param coreConfId: (optional) id of the core conformation to use
    #     :param randomSeed: (optional) seed for the random number generator
    #     :param getForceField: (optional) force field to use for the optimization of molecules
    #     :return: RDKit molecule object containing the embedded conformations.
    #     """

    #     match = mol.GetSubstructMatch(core)
    #     if not match:
    #         raise ValueError("molecule doesn't match the core")
    #     coordMap = {}
    #     coreConf = core.GetConformer(coreConfId)
    #     for i, idxI in enumerate(match):
    #         corePtI = coreConf.GetAtomPosition(i)
    #         coordMap[idxI] = corePtI

    #     cids = EmbedMultipleConfs(mol, numConfs=numConfs, coordMap=coordMap, randomSeed=randomSeed, **kwargs)
    #     cids = list(cids)
    #     if len(cids) == 0:
    #         raise ValueError('Could not embed molecule.')

    #     algMap = [(j, i) for i, j in enumerate(match)]

    #     energies = []
        
    #     if not useTethers:
    #         # clean up the conformation
    #         for cid in cids:
    #             ff = getForceField(mol, confId=cid)
    #             for i, idxI in enumerate(match):
    #                 for j in range(i + 1, len(match)):
    #                     idxJ = match[j]
    #                     d = coordMap[idxI].Distance(coordMap[idxJ])
    #                     ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.)
    #             ff.Initialize()
    #             n = 4
    #             more = ff.Minimize()
    #             while more and n:
    #                 more = ff.Minimize()
    #                 n -= 1
    #             # rotate the embedded conformation onto the core:
    #             rms = AlignMol(mol, core, atomMap=algMap)
    #     else:
    #         # rotate the embedded conformation onto the core:
    #         for cid in cids:
    #             rms = AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    #             ff = getForceField(mol, confId=cid)
    #             conf = core.GetConformer()
    #             for i in range(core.GetNumAtoms()):
    #                 p = conf.GetAtomPosition(i)
    #                 pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
    #                 ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.)
    #             ff.Initialize()
    #             n = 4
    #             more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
    #             while more and n:
    #                 more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
    #                 n -= 1
    #             # realign
    #             rms = AlignMol(mol, core, prbCid=cid, atomMap=algMap)
    #             energy = ff.CalcEnergy()
    #             energies.append(energy)
    #     return mol, energies

    # def add_charge(self, conformers_path):

    #     confs = sorted(glob(os.path.join(conformers_path + '*', 'conformers_*.mol2')))
    #     for conf in confs:

    #         # for debug
    #         # obabel -ipdb ${input}.pdb -opdb -O ${output}.pdb -ph 7.4
    #         if conf == '/work/ChemTSv2/work/results/lead_000/conformers_000.mol2':
    #             subprocess.run(['cp', conf, conf.replace('.mol2','_org.mol2')])

    #         subprocess.run(['obabel', '-imol2', conf,'-omol2', '-O', conf, '-ph', '7.4'])

    #         # rdkitではmol2扱いにくいのでpdbにしておく 
    #         subprocess.run(['obabel', '-imol2', conf,'-opdb', '-O', conf.replace('mol2','pdb')])

