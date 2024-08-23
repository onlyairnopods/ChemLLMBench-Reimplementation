import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import datetime
from typing import List, Any, Literal, Optional, Union, Tuple, Sequence, Callable
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.debug')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

from .utils import top_n_scaffold_similar_molecules, top_n_similar_strings, get_scaffold_fp, canonicalize_smiles
from .pkg_mol2cap_cap2mol.create_prompt import get_input_output_columns_by_task, create_few_shot_prompt, create_zero_shot_prompt
from .pkg_mol2cap_cap2mol.metric import evaluate_one_molecule_design, evaluate_one_molecule_captioning

PARAMS = [
    # sample_method, sample_num, mol_format
    ('Scaffold_SIM', 10, 'SMILES'),
    ('Scaffold_SIM', 5, 'SMILES'),
    ('Random', 10, 'SMILES'),
]

def _load_datasets(dataset_file_path: str, set: Literal['train', 'test']) -> pd.DataFrame:
    df = pd.read_csv(dataset_file_path)
    df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
    if set == 'train':
        df['scaffold_fp'] = df['SMILES'].apply(lambda x: get_scaffold_fp(x))
    elif set == 'test':
        df = df.reset_index()
    return df

def _few_shot_merge_performance(
        result_folder: str,
        model_name: str,
        task: str,
) -> None:
    performace_results = []
    caption_cols = [
        'task', 'model', 'mol_format', 'sample_num', 'sample_method',
    ]

    for sample_method, sample_num, mol_format in PARAMS:
        predicted_details_file = result_folder + f"few_shot_test_predicted_details_{task}_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"
        predicted_details = pd.read_csv(predicted_details_file)

        if task == 'molecule_design':
            metrics, stds = evaluate_one_molecule_design(predicted_details)
            performace_results.append([task, model_name, mol_format, sample_num, sample_method] + metrics + stds)
        
            metrics_cols = [
                'bleu', 
                'exact_match', 
                'levenshtein', 
                'validity', 
                'maccs_sims', 
                'rdk_sims', 
                'morgan_sims', 
                'fcd'
            ]

            std_cols = [i + "_std" for i in metrics_cols]

        elif task == 'molecule_captioning':
            metrics, stds = evaluate_one_molecule_captioning(predicted_details)
            performace_results.append([task, model_name, mol_format, sample_num, sample_method] + metrics + stds)

            metrics_cols = [
                'bleu2', 'bleu4', 'rouge1', 'rouge2', 'rougel', 'meteor', 
            ]

            std_cols = [i + "_std" for i in metrics_cols]
        
        performace_results_df = pd.DataFrame(performace_results, columns=caption_cols + metrics_cols + std_cols)
        performace_results_df_file = result_folder + f"few_shot_test_{task}_all_performance_{model_name}.csv"
        performace_results_df.to_csv(performace_results_df_file, index=False)
        print(f"{'='*30} Evaluation results saved to {performace_results_df_file} {'='*30}")

def _zero_shot_merge_performance(
        result_folder: str,
        model_name: str,
        task: str,
) -> None:
    performace_results = []

    predicted_details_file = result_folder + f"zero_shot_test_predicted_details_{task}_{model_name}.csv"
    predicted_details = pd.read_csv(predicted_details_file)

    caption_cols = [
        'task', 'model',
    ]

    if task == 'molecule_design':
        metrics, stds = evaluate_one_molecule_design(predicted_details)
        performace_results.append([task, model_name] + metrics + stds)
    
        metrics_cols = [
            'bleu', 
            'exact_match', 
            'levenshtein', 
            'validity', 
            'maccs_sims', 
            'rdk_sims', 
            'morgan_sims', 
            'fcd'
        ]

        std_cols = [i + "_std" for i in metrics_cols]

    elif task == 'molecule_captioning':
        metrics, stds = evaluate_one_molecule_captioning(predicted_details)
        performace_results.append([task, model_name] + metrics + stds)

        metrics_cols = [
            'bleu2', 'bleu4', 'rouge1', 'rouge2', 'rougel', 'meteor', 
        ]

        std_cols = [i + "_std" for i in metrics_cols]
    
    performace_results_df = pd.DataFrame(performace_results, columns=caption_cols + metrics_cols + std_cols)
    performace_results_df_file = result_folder + f"zero_shot_test_{task}_all_performance_{model_name}.csv"
    performace_results_df.to_csv(performace_results_df_file, index=False)
    print(f"{'='*30} Evaluation results saved to {performace_results_df_file} {'='*30}")


def mol2cap_cap2mol_few_shot(
        train_dataset_file_path: str,
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
        task: Literal['molecule_captioning', 'molecule_design']
) -> None:
    
    """
    Arguments:
        - train_dataset_file_path: the csv file path of the training dataset
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - task: 'molecule_captioning' or 'molecule_design'
    """
    
    train_data = _load_datasets(train_dataset_file_path, set='train')
    test_data = _load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    
    for sample_method, sample_num, mol_format in PARAMS:

        prompts_log_file = result_folder + f"fs_{task}_{model_name}_{mol_format}_{sample_num}_{sample_method}.log"
        performance_file = result_folder + f"fs_performance_{task}_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"
        predicted_details_file = result_folder + f"fs_predicted_details_{task}_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"

        if os.path.exists(performance_file):
            print(f"{performance_file} exits, continue to next params")
            continue

        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(prompts_log_file, "a") as file:
            file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

        # store
        predicted_details = []
        predicted_outputs = []

        # restore previous data and skip if exists
        previous_index = []
        if os.path.exists(predicted_details_file):
            previous = pd.read_csv(predicted_details_file)
            previous_index = list(previous['index'])

            predicted_details = previous.values.tolist()
            predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

        # load column fields according to task
        input_col, output_col = get_input_output_columns_by_task(task)

        for idx, row in tqdm(test_data.iterrows()):
            input = row[input_col]
            output = row[output_col]
            index = row['index']

            if index in previous_index:
                continue

            # sample ICL examples
            if sample_method == 'Random':
                chunk = train_data.sample(sample_num, random_state=42)
            elif sample_method == 'Scaffold_SIM':
                if input_col == 'SMILES':
                    top_smiles = top_n_scaffold_similar_molecules(input, list(train_data['scaffold_fp']), list(train_data['SMILES']), top_n=sample_num)
                else:
                    top_smiles = top_n_similar_strings(input, list(train_data[input_col]), top_n=sample_num)
                chunk = train_data[train_data[input_col].isin(top_smiles)]

            sample_examples = list(zip(chunk[input_col].values, chunk[output_col].values))

            # build prompt and save
            prompt = create_few_shot_prompt(input, sample_examples, task)
            with open(prompts_log_file, "a") as file:
                file.write(prompt + "\n")
                file.write("=" * 50 + "\n")

            try:
                # model inference
                predicted_output = model(prompt)
            except Exception as e:
                print(f"Error in model inference: {e}")

            # TODO: reduce prompt if it is too long

            predicted_outputs.append(predicted_output)
            predicted_details.append([index, input] + [output] + predicted_output)
        
        details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
        details_df.to_csv(predicted_details_file, index=False)

    _few_shot_merge_performance(result_folder, model_name, task)


def mol2cap_cap2mol_zero_shot(
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
        task: Literal['molecule_captioning', 'molecule_design']
) -> None:
    
    """
    Arguments:
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - task: 'molecule_captioning' or 'molecule_design'
    """
    
    test_data = _load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)

    prompts_log_file = result_folder + f"zs_{task}_{model_name}.log"
    performance_file = result_folder + f"zs_performance_{task}_{model_name}.csv"
    predicted_details_file = result_folder + f"zs_predicted_details_{task}_{model_name}.csv"

    if os.path.exists(performance_file):
        print(f"{performance_file} exits, continue to next params")
        raise RuntimeError("exist")

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(prompts_log_file, "a") as file:
        file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

    # store
    predicted_details = []
    predicted_outputs = []

    # restore previous data and skip if exists
    previous_index = []
    if os.path.exists(predicted_details_file):
        previous = pd.read_csv(predicted_details_file)
        previous_index = list(previous['index'])

        predicted_details = previous.values.tolist()
        predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

    # load column fields according to task
    input_col, output_col = get_input_output_columns_by_task(task)

    for idx, row in tqdm(test_data.iterrows()):
        input = row[input_col]
        output = row[output_col]
        index = row['index']

        if index in previous_index:
            continue

        # build prompt and save
        prompt = create_zero_shot_prompt(input, task)
        with open(prompts_log_file, "a") as file:
            file.write(prompt + "\n")
            file.write("=" * 50 + "\n")

        # model inference
        predicted_output = model(prompt)

        predicted_outputs.append(predicted_output)
        predicted_details.append([index, input] + [output] + predicted_output)
    
    details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    details_df.to_csv(predicted_details_file, index=False)

    _zero_shot_merge_performance(result_folder, model_name, task)