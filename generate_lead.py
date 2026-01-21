import logging
import os
import shutil
import glob
import subprocess
import sys
import copy
from pathlib import Path
from typing import List

import pandas as pd
import yaml
import rdkit
from rdkit import Chem

import ChemTSv2.chemts_methods as cm

cwd = Path(__file__).resolve().parent
# target_dirname = 'work/results'

class Generate_Lead:
    def __init__(self, config, log_file):
        self.input_compound_files = []
        self.conf = config  
        self.out_log_file = log_file
        self.logger = cm.setup_custom_logger('ChemTS', str(self.out_log_file))
        self.generation_workflow = Path(self.conf['GENERATE_WORKFLOW']['working_directory'])
        self.target_dirname = Path(self.conf['ChemTS']['target_dirname'])

    def run(self, trajectory_dirs: List[Path]) -> List[Path]:
        rank_output_dirs = []
        for trajectory_dir in trajectory_dirs:
            self.logger.info(str(trajectory_dir))
            sincho_result_file = trajectory_dir / 'sincho_result.yaml'

            # TODO: AA_Score_Calculation.pyの_parse_trajectory_nameと同じ処理
            trajectory_name = trajectory_dir.name
            trajectory_num = trajectory_name.split('_')[-1]

            ChemTS_output_dir = self.generation_workflow / 'ChemTS'
            trajectory_output_dir = ChemTS_output_dir / trajectory_name
            trajectory_output_dir.mkdir(parents=True, exist_ok = True)
            
            with open(sincho_result_file, 'r')as f:
                sincho_results = yaml.safe_load(f)
            sincho_results = sincho_results['SINCHO_result']
            
            input_compound_file = trajectory_dir / f'lig_{trajectory_num}.pdb'
            self.input_compound_files.append(input_compound_file)
            input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(str(input_compound_file)))

            # 中性ならTrue,電荷ありならFalse
            self.is_neutral = cm.check_neutral(input_compound_smiles)
            
            for rank, sincho_result in sincho_results.items():
                self.logger.info(f"rank , {rank}")
                rank_output_dir = trajectory_output_dir / rank
                rank_output_dir.mkdir(parents=True, exist_ok = True)
                rank_output_dirs.append(rank_output_dir)

                # 生やしたい分子量を取得
                estimate_add_mw = sincho_result['mw']
                weight_model_dir = cm.select_weight_model(input_compound_smiles, estimate_add_mw)
                self.logger.info(f"weight_model_dir , {weight_model_dir}")
                
                extend_atom = sincho_result['atom_num'].split('.')[1].split('_')[-1]

                if not self.is_neutral:
                    # 電荷ありをopenbabelで中性化する
                    # SINCHOは電荷ありでやってる
                    # 中性化→プロパティ計算→SMILES並び替え
                    # lig_000.pdb -> lig_000_org.pdbとして保持し、中性化したものをlig_000.pdbとする
                    # pdbだと上手くいかないからmol2経由する lig_000.pdb -> lig_000.mol2 -> (neutral) -> lig_000.pdb(同名だが中性化されている)
                    
                    # TODO: `input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(str(input_compound_file)))`のところで中性化すれば良いと思う。(要確認)
                    self.logger.info('ligand has charges.')
                    # openbabelで中性化
                    cm.do_neutral(str(input_compound_file), self.logger)
                    # SMILESを中性化に更新
                    input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(str(input_compound_file)))

                # 初期SMILESの物性値を計算し、configに記載しておく
                properties = cm.calculate_compound_properties(input_compound_smiles)
                local_config = copy.deepcopy(self.conf)
                local_config['ChemTS'].update(properties)

                # SMILESの並び替え(中性化→計算→並び替えの順序は保持する)
                rearrange_smi = cm.set_rearrange_smiles(str(input_compound_file), extend_atom, logger = self.logger)
                self.logger.info(f"smi , {input_compound_smiles}")

                # 不正SMILESのチェック(現状は[n]のみ)
                if not cm.check_error_smiles(rearrange_smi):
                    rearrange_smi = cm.modify_smiles(rearrange_smi, str(input_compound_file.parent), logger = self.logger)

                self.logger.info(f"rearrange_smi , {rearrange_smi}")

                (cwd / 'work').mkdir(parents=True, exist_ok=True)
                setting_yaml_path = cwd / 'work' / '_setting.yaml'
                if setting_yaml_path.exists():
                    setting_yaml_path.unlink()
                cm.make_config_file({**local_config, **sincho_result}, weight_model_dir, os.path.join('ChemTSv2', 'work', '_setting.yaml'))

                # 化合物生成をn回
                df_result_list = []
                for n in range(1, int(local_config['ChemTS']['num_chemts_loops'])+1):
                    df_result_one_cycle = self._run_chemts_process(n, rearrange_smi, cwd)
                    df_result_list.append(df_result_one_cycle)
                
                df_result_all = pd.concat(df_result_list, ignore_index=True) if df_result_list else pd.DataFrame()
                        
                # for debug df_result_all
                # df_result_all = pd.read_csv(os.path.join(cwd, self.target_dirname, 'results.csv'))
                
                # n回分を一つのファイルにし、個々のファイルは消しておく
                output_csv_path = cwd / self.target_dirname / 'results.csv'
                df_result_all.to_csv(str(output_csv_path))
                result_csv_path = cwd / self.target_dirname / 'result.csv'
                if result_csv_path.exists():
                    result_csv_path.unlink()
                
                # 今回の生成のrewardなどをプロット
                cm.plot_reward(str(output_csv_path))

                source_dir = cwd / self.target_dirname
                for file_path in source_dir.glob('*'):
                    shutil.move(str(file_path), str(rank_output_dir))
        return rank_output_dirs

    def _run_chemts_process(self, n, rearrange_smi, cwd) -> pd.DataFrame:
        setting_file = cwd / 'work' / '_setting.yaml'
        with open(self.out_log_file, 'a') as stdout_f:
            try:
                cmd = [ 'python', 'run.py', '-c', str(setting_file), '--input_smiles', rearrange_smi ]
                subprocess.run(cmd, cwd=str(cwd), stdout=stdout_f, stderr=stdout_f, check=True)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"ChemTS execution failed in trial {n}: {e}")
                raise
            
        result_dir = cwd / self.target_dirname
        # mv result_C* -> result.csv
        pattern = 'result_C*'
        matched_files = list(result_dir.glob(pattern))
        
        if len(matched_files) == 1:
            shutil.move(str(matched_files[0]), str(result_dir / 'result.csv'))
        elif len(matched_files) > 1:
            raise RuntimeError(f"Multiple result files found: {matched_files}. Expected only one.")
        else:
            self.logger.warning("No result file found matching 'result_C*'")

        df_result_one_cycle = pd.read_csv(str(result_dir / 'result.csv'))
        df_result_one_cycle.insert(0, 'trial', n) 
        
        run_log_path = result_dir / 'run.log'
        run_log_all_path = result_dir / 'run.log.all'
        if run_log_path.exists():
            with open(run_log_path, 'r') as f_in, open(run_log_all_path, 'a') as f_out:
                f_out.write(f_in.read())
    
        return df_result_one_cycle