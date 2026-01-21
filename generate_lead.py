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

CHEMTS_CURRENT_DIR = Path(__file__).resolve().parent

class Generate_Lead:
    def __init__(self, config, log_file):
        self.input_compound_files = []
        self.base_chemts_config = copy.deepcopy(config['ChemTS'])
        self.out_log_file = log_file
        self.logger = cm.setup_custom_logger('ChemTS', str(self.out_log_file))
        self.generation_workflow = Path(config['GENERATE_WORKFLOW']['working_directory'])
        self.num_chemts_loops = int(config['ChemTS']['num_chemts_loops'])

    def run(self, trajectory_dirs: List[Path]) -> List[Path]:
        rank_output_dirs = []
        for trajectory_dir in trajectory_dirs:
            self.logger.info(str(trajectory_dir))
            sincho_result_file = trajectory_dir / 'sincho_result.yaml'
            with open(sincho_result_file, 'r')as f:
                sincho_results = yaml.safe_load(f)
            sincho_results = sincho_results['SINCHO_result']

            # TODO: AA_Score_Calculation.pyの_parse_trajectory_nameと同じ処理
            trajectory_name = trajectory_dir.name
            trajectory_num = trajectory_name.split('_')[-1]

            ChemTS_output_dir = self.generation_workflow / 'ChemTS'
            trajectory_output_dir = ChemTS_output_dir / trajectory_name
            trajectory_output_dir.mkdir(parents=True, exist_ok = True)
            
            
            input_compound_file = trajectory_dir / f'lig_{trajectory_num}.pdb'
            self.input_compound_files.append(input_compound_file)
            input_compound_smiles = Chem.MolToSmiles(Chem.MolFromPDBFile(str(input_compound_file)))

            # 中性ならTrue,電荷ありならFalse
            self.is_neutral = cm.check_neutral(input_compound_smiles)
            
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

            for rank, sincho_result in sincho_results.items():
                self.logger.info(f"rank , {rank}")
                rank_output_dir = trajectory_output_dir / rank
                rank_output_dir.mkdir(parents=True, exist_ok = True)
                rank_output_dirs.append(rank_output_dir)

                # 生やしたい分子量を取得
                estimate_add_mw = sincho_result['mw']
                weight_model_dir = cm.select_weight_model(input_compound_smiles, estimate_add_mw)
                self.logger.info(f"weight_model_dir , {weight_model_dir}")
                
                # 初期SMILESの物性値を計算し、configに記載しておく
                properties = cm.calculate_compound_properties(input_compound_smiles)
                local_config = copy.deepcopy(self.base_chemts_config)
                local_config.update(properties)

                # SMILESの並び替え(中性化→計算→並び替えの順序は保持する)
                extend_atom = sincho_result['atom_num'].split('.')[1].split('_')[-1]
                rearrange_smi = cm.set_rearrange_smiles(str(input_compound_file), extend_atom, logger = self.logger)
                self.logger.info(f"smi , {input_compound_smiles}")

                # 不正SMILESのチェック(現状は[n]のみ)
                if not cm.check_error_smiles(rearrange_smi):
                    rearrange_smi = cm.modify_smiles(rearrange_smi, str(input_compound_file.parent), logger = self.logger)

                self.logger.info(f"rearrange_smi , {rearrange_smi}")

                # 化合物生成をn回
                working_dirs = []
                for n in range(1, self.num_chemts_loops+1):
                    working_dir = rank_output_dir / 'working' / f'trial_{n}'
                    working_dir.mkdir(parents=True, exist_ok=True)
                    setting_file_name = '_setting.yaml'
                    
                    local_config['output_dir'] = str(working_dir)
                    cm.create_config_file(local_config, sincho_result, weight_model_dir, str(working_dir / setting_file_name), logger = self.logger)

                    self._run_chemts_process(n, rearrange_smi, working_dir, setting_file_name)
                    working_dirs.append(working_dir)
                
                # ログの集約
                with open(rank_output_dir / 'run.log.all', 'w') as f_out:
                    for working_dir in working_dirs:
                        run_log_path = working_dir / 'run.log'
                        if run_log_path.exists():
                            with open(run_log_path, 'r') as f_in:
                                f_out.write(f_in.read())

                # 1. 結果CSVの集約
                df_result_list = []
                for n, working_dir in enumerate(working_dirs, start=1):
                    matched_files = list(working_dir.glob('result_C*'))
                    if len(matched_files) == 1:
                        df = pd.read_csv(str(matched_files[0]))
                        df.insert(0, 'trial', n)
                        df_result_list.append(df)
                    elif len(matched_files) > 1:
                        raise RuntimeError(f"Multiple result files found in {working_dir}: {matched_files}. Expected only one.")
                    else:
                        raise RuntimeError(f"No result file found matching 'result_C*' in {working_dir}")

                df_result_all = pd.concat(df_result_list, ignore_index=True) if df_result_list else pd.DataFrame()
                
                # n回分を一つのファイルに集約する
                output_csv_path = rank_output_dir / 'results.csv'
                df_result_all.to_csv(str(output_csv_path))
                
                # 今回の生成のrewardなどをプロット
                cm.plot_reward(str(output_csv_path))

        return rank_output_dirs

    def _run_chemts_process(self, trial_n, rearrange_smi, working_dir: Path, setting_file_name: str) -> None:
        with open(self.out_log_file, 'a') as stdout_f:
            try:
                cmd = [ 'python', 'run.py', '-c', str(working_dir / setting_file_name), '--input_smiles', rearrange_smi ]
                subprocess.run(cmd, cwd=str(CHEMTS_CURRENT_DIR), stdout=stdout_f, stderr=stdout_f, check=True)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"ChemTS execution failed in trial {trial_n}: {e}")
                raise