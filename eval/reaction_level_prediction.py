import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import datetime
import ast
from typing import List, Any, Literal, Optional, Union, Tuple, Sequence, Callable
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.debug')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

from .utils import top_n_scaffold_similar_molecules, canonicalize_smiles, get_scaffold_fp
from .pkg_reaction_level_prediction.create_prompt import create_few_shot_prompt_reaction_prediction, create_zero_shot_prompt_reaction_prediction, create_few_shot_prompt_retrosynthesis, create_zero_shot_prompt_retrosynthesis, create_few_shot_prompt_yield_prediction, create_zero_shot_prompt_yield_prediction

PARAMS = [
    # sample_method, sample_num, mol_format
    ('Scaffold_SIM', 20, 'SMILES'),
    ('Scaffold_SIM', 5, 'SMILES'),
    ('Random', 20, 'SMILES'),
]

YIELD_PARAMS = [
    # sample_method, sample_num, mol_format
    ('Random', 4, 'SMILES'),
    ('Random', 8, 'SMILES'),
]

def _reaction_prediction_load_datasets(dataset_file_path: str, set: Literal['train', 'test']) -> pd.DataFrame:
    df = pd.read_csv(dataset_file_path)
    if set == 'train':
        df['scaffold_fp'] = df['reactant'].apply(get_scaffold_fp)
    elif set == 'test':
        df = df.reset_index()
    return df

def reaction_prediction_few_shot(
        train_dataset_file_path: str,
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
) -> None:
    
    """
    Arguments:
        - train_dataset_file_path: the csv file path of the training dataset
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
    """

    train_data = _reaction_prediction_load_datasets(train_dataset_file_path, set='train')
    test_data = _reaction_prediction_load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    input_col = 'reactant'
    output_col = 'product'

    for sample_method, sample_num, mol_format in PARAMS:

        prompts_log_file = result_folder + f"fs_reaction_prediction_{model_name}_{mol_format}_{sample_num}_{sample_method}.log"
        performance_file = result_folder + f"fs_performance_reaction_prediction_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"
        predicted_details_file = result_folder + f"fs_predicted_details_reaction_prediction_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"

        if os.path.exists(performance_file):
            print(f"{performance_file} exits, continue to next params")
            continue

        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(prompts_log_file, "a") as file:
            file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

        # store
        predicted_details = []
        performance_results = []
        predicted_outputs = []
    
        # restore previous data and skip if exists
        previous_index = []
        if os.path.exists(predicted_details_file):
            previous = pd.read_csv(predicted_details_file)
            previous_index = list(previous['index'])

            predicted_details = previous.values.tolist()
            predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

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
                top_smiles = top_n_scaffold_similar_molecules(input, list(train_data['scaffold_fp']), list(train_data[input_col]), top_n=sample_num)
                chunk = train_data[train_data[input_col].isin(top_smiles)]

            sample_examples = list(zip(chunk[input_col].values, chunk[output_col].values))

            # build prompt and save
            prompt = create_few_shot_prompt_reaction_prediction(input, sample_examples)
            with open(prompts_log_file, "a") as file:
                file.write(prompt + "\n")
                file.write("=" * 50 + "\n")

            # model inference
            predicted_output = model(prompt)

            predicted_outputs.append(predicted_output)
            predicted_details.append([index, input] + [output] + predicted_output)

        details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
        details_df.to_csv(predicted_details_file, index=False)

        # evaluate
        acc_list = []
        # each evaluation experiment is repeated five times and the mean and variance are reported.
        repeat_times = 5
        for repeat in range(repeat_times):
            tpredicted_outputs = [i[repeat] for i in predicted_outputs]
            correct = 0
            all_num = len(test_data)
            for idx, gt in enumerate(list(test_data[output_col])):
                pred = tpredicted_outputs[idx]
                try:
                    mol = Chem.MolFromSmiles(pred)
                    pred = Chem.MolToSmiles(mol)
                except Exception as e:
                    continue

                pred_list = pred.split(".")
                if gt in pred_list:
                    correct += 1
            acc = correct / all_num
            acc_list.append(acc)
        
        # save results to file
        performance_results.append([model_name, mol_format, sample_num, sample_method, np.mean(acc_list)] + acc_list)

        tem = pd.DataFrame(performance_results, columns=['model_name', 'mol_format', 'sample_num', 'sample_method', 'avg_metric'] \
                        + [f'metric_{i}' for i in range(repeat_times)])
        tem.to_csv(performance_file, index=False)

def reaction_prediction_zero_shot(
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
) -> None:
    
    """
    Arguments:
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
    """

    test_data = _reaction_prediction_load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    input_col = 'reactant'
    output_col = 'product'

    prompts_log_file = result_folder + f"zs_reaction_prediction_{model_name}.log"
    performance_file = result_folder + f"zs_performance_reaction_prediction_{model_name}.csv"
    predicted_details_file = result_folder + f"zs_predicted_details_reaction_prediction_{model_name}.csv"

    if os.path.exists(performance_file):
        print(f"{performance_file} exits, continue to next params")
        raise RuntimeError("exist")

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(prompts_log_file, "a") as file:
        file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

    # store
    predicted_details = []
    performance_results = []
    predicted_outputs = []

    # restore previous data and skip if exists
    previous_index = []
    if os.path.exists(predicted_details_file):
        previous = pd.read_csv(predicted_details_file)
        previous_index = list(previous['index'])

        predicted_details = previous.values.tolist()
        predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

    for idx, row in tqdm(test_data.iterrows()):
        input = row[input_col]
        output = row[output_col]
        index = row['index']

        if index in previous_index:
            continue

        # build prompt and save
        prompt = create_zero_shot_prompt_reaction_prediction(input)
        with open(prompts_log_file, "a") as file:
            file.write(prompt + "\n")
            file.write("=" * 50 + "\n")

        # model inference
        predicted_output = model(prompt)

        predicted_outputs.append(predicted_output)
        predicted_details.append([index, input] + [output] + predicted_output)

    details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    details_df.to_csv(predicted_details_file, index=False)

    # evaluate
    acc_list = []
    # each evaluation experiment is repeated five times and the mean and variance are reported.
    repeat_times = 5
    for repeat in range(repeat_times):
        tpredicted_outputs = [i[repeat] for i in predicted_outputs]
        correct = 0
        all_num = len(test_data)
        for idx, gt in enumerate(list(test_data[output_col])):
            pred = tpredicted_outputs[idx]
            try:
                mol = Chem.MolFromSmiles(pred)
                pred = Chem.MolToSmiles(mol)
            except Exception as e:
                continue

            pred_list = pred.split(".")
            if gt in pred_list:
                correct += 1
        acc = correct / all_num
        acc_list.append(acc)
    
    # save results to file
    performance_results.append([model_name, np.mean(acc_list)] + acc_list)

    tem = pd.DataFrame(performance_results, columns=['model_name', 'avg_metric'] \
                    + [f'metric_{i}' for i in range(repeat_times)])
    tem.to_csv(performance_file, index=False)

###################################################################################

def _reagents_selection_load_datasets(dataset_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_file_path) # contains 4 columns: task, candidate_rank, candidate_over_30_yield, yield_details
    for i in df.columns[1:]:
        df[i] = df[i].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x) # convert str to list
    df = df.reset_index()
    return df

def reagents_selection(
        dataset_file_path: str,
        result_folder: str,
        model_name: str,
        model: Callable[[str], str],
        task: Literal['ligand', 'reactant', 'solvent']
) -> None:
    
    """
    Only zero-shot method.

    Arguments:
        - dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - task: one of the ['ligand', 'reactant', 'solvent']
    """

    assert task in ['ligand', 'reactant', 'solvent'], f"In reagents selection, task should be in ['ligand', 'reactant', 'solvent']"
    test_data = _reagents_selection_load_datasets(dataset_file_path)
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
    performance_results = []
    predicted_outputs = []

    # restore previous data and skip if exists
    previous_index = []
    if os.path.exists(predicted_details_file):
        previous = pd.read_csv(predicted_details_file)
        previous_index = list(previous['index'])

        predicted_details = previous.values.tolist()
        predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

    for idx, row in tqdm(test_data.iterrows()):
        input = row['task']
        output = row['candidate_rank'][0]
        index = row['index']

        if index in previous_index:
            continue

        # build prompt and save
        prompt = input
        with open(prompts_log_file, "a") as file:
            file.write(prompt + "\n")
            file.write("=" * 50 + "\n")

        # model inference
        predicted_output = model(prompt)

        predicted_outputs.append(predicted_output)
        predicted_details.append([index, input] + [output] + predicted_output)

    details_df = pd.DataFrame(predicted_details, columns=['index', 'task', 'candidate_rank_1', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    details_df.to_csv(predicted_details_file, index=False)

    # evaluate
    acc_list = []
    # each evaluation experiment is repeated five times and the mean and variance are reported.
    repeat_times = 5
    for repeat in range(repeat_times):
        tpredicted_outputs = [i[repeat] for i in predicted_outputs]
        correct = 0
        all_num = len(test_data)
        for idx, gt in enumerate(list(test_data['candidate_rank'])):
            pred = tpredicted_outputs[idx]
            try:
                mol = Chem.MolFromSmiles(pred)
                pred = Chem.MolToSmiles(mol)
            except Exception as e:
                continue

            if task == 'reactant' or task == 'solvent':
                if pred == gt[0]: # top-1 accuracy
                    correct += 1
            elif task == 'ligand': # top-50% accuracy
                if pred in gt[:int(len(gt) * 0.5)]:
                    correct += 1

        acc = correct / all_num
        acc_list.append(acc)
    
    # save results to file
    performance_results.append([task, model_name, np.mean(acc_list)] + acc_list)

    tem = pd.DataFrame(performance_results, columns=['task', 'model_name', 'avg_metric'] \
                    + [f'metric_{i}' for i in range(repeat_times)])
    tem.to_csv(performance_file, index=False)

###################################################################################

def _retrosynthesis_load_datasets(dataset_file_path: str, set: Literal['train', 'test']) -> pd.DataFrame:
    df = pd.read_csv(dataset_file_path)
    if set == 'train':
        df['scaffold_fp'] = df['products_smiles'].apply(lambda x: get_scaffold_fp(x))
    elif set == 'test':
        df = df.reset_index()
    return df

def retrosynthesis_few_shot(
        train_dataset_file_path: str,
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
) -> None:
    
    """
    Arguments:
        - train_dataset_file_path: the csv file path of the training dataset
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
    """

    train_data = _retrosynthesis_load_datasets(train_dataset_file_path, set='train')
    test_data = _retrosynthesis_load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    input_col = 'products_smiles'
    output_col = 'reactants_smiles'

    for sample_method, sample_num, mol_format in PARAMS:

        prompts_log_file = result_folder + f"fs_retrosynthesis_{model_name}_{mol_format}_{sample_num}_{sample_method}.log"
        performance_file = result_folder + f"fs_performance_retrosynthesis_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"
        predicted_details_file = result_folder + f"fs_predicted_details_retrosynthesis_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"

        if os.path.exists(performance_file):
            print(f"{performance_file} exits, continue to next params")
            continue

        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(prompts_log_file, "a") as file:
            file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

        # store
        predicted_details = []
        performance_results = []
        predicted_outputs = []
    
        # restore previous data and skip if exists
        previous_index = []
        if os.path.exists(predicted_details_file):
            previous = pd.read_csv(predicted_details_file)
            previous_index = list(previous['index'])

            predicted_details = previous.values.tolist()
            predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

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
                top_smiles = top_n_scaffold_similar_molecules(input, list(train_data['scaffold_fp']), list(train_data[input_col]), top_n=sample_num)
                chunk = train_data[train_data[input_col].isin(top_smiles)]

            sample_examples = list(zip(chunk[input_col].values, chunk[output_col].values))

            # build prompt and save
            prompt = create_few_shot_prompt_retrosynthesis(input, sample_examples)
            with open(prompts_log_file, "a") as file:
                file.write(prompt + "\n")
                file.write("=" * 50 + "\n")

            # model inference
            predicted_output = model(prompt)

            predicted_outputs.append(predicted_output)
            predicted_details.append([index, input] + [output] + predicted_output)

        details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
        details_df.to_csv(predicted_details_file, index=False)

        # evaluate
        acc_list = []
        # each evaluation experiment is repeated five times and the mean and variance are reported.
        repeat_times = 5
        for repeat in range(repeat_times):
            tpredicted_outputs = [i[repeat] for i in predicted_outputs]
            correct = 0
            all_num = len(test_data)
            for idx, gt in enumerate(list(test_data[output_col])):
                pred = tpredicted_outputs[idx]
                pred_list_sorted = sorted(pred.split('.'))
                gt_list_sorted = sorted(gt[0].split('.'))
                if pred_list_sorted == gt_list_sorted:
                    correct += 1

            acc = correct / all_num
            acc_list.append(acc)
        
        # save results to file
        performance_results.append([model_name, mol_format, sample_num, sample_method, np.mean(acc_list)] + acc_list)

        tem = pd.DataFrame(performance_results, columns=['model_name', 'mol_format', 'sample_num', 'sample_method', 'avg_metric'] \
                        + [f'metric_{i}' for i in range(repeat_times)])
        tem.to_csv(performance_file, index=False)

def retrosynthesis_zero_shot(
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
) -> None:

    """
    Arguments:
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
    """

    test_data = _retrosynthesis_load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    input_col = 'products_smiles'
    output_col = 'reactants_smiles'

    prompts_log_file = result_folder + f"zs_retrosynthesis_{model_name}.log"
    performance_file = result_folder + f"zs_performance_retrosynthesis_{model_name}.csv"
    predicted_details_file = result_folder + f"zs_predicted_details_retrosynthesis_{model_name}.csv"

    if os.path.exists(performance_file):
        print(f"{performance_file} exits, continue to next params")
        raise RuntimeError("exist")

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(prompts_log_file, "a") as file:
        file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

    # store
    predicted_details = []
    performance_results = []
    predicted_outputs = []

    # restore previous data and skip if exists
    previous_index = []
    if os.path.exists(predicted_details_file):
        previous = pd.read_csv(predicted_details_file)
        previous_index = list(previous['index'])

        predicted_details = previous.values.tolist()
        predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

    for idx, row in tqdm(test_data.iterrows()):
        input = row[input_col]
        output = row[output_col]
        index = row['index']

        if index in previous_index:
            continue

        # build prompt and save
        prompt = create_zero_shot_prompt_retrosynthesis(input)
        with open(prompts_log_file, "a") as file:
            file.write(prompt + "\n")
            file.write("=" * 50 + "\n")

        # model inference
        predicted_output = model(prompt)

        predicted_outputs.append(predicted_output)
        predicted_details.append([index, input] + [output] + predicted_output)

    details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    details_df.to_csv(predicted_details_file, index=False)

    # evaluate
    acc_list = []
    # each evaluation experiment is repeated five times and the mean and variance are reported.
    repeat_times = 5
    for repeat in range(repeat_times):
        tpredicted_outputs = [i[repeat] for i in predicted_outputs]
        correct = 0
        all_num = len(test_data)
        for idx, gt in enumerate(list(test_data[output_col])):
            pred = tpredicted_outputs[idx]
            pred_list_sorted = sorted(pred.split('.'))
            gt_list_sorted = sorted(gt[0].split('.'))
            if pred_list_sorted == gt_list_sorted:
                correct += 1

        acc = correct / all_num
        acc_list.append(acc)
    
    # save results to file
    performance_results.append([model_name, np.mean(acc_list)] + acc_list)

    tem = pd.DataFrame(performance_results, columns=['model_name', 'avg_metric'] \
                    + [f'metric_{i}' for i in range(repeat_times)])
    tem.to_csv(performance_file, index=False)

###################################################################################

def _yield_prediction_load_datasets(dataset_file_path: str, set: Literal['train', 'test']) -> pd.DataFrame:
    df = pd.read_csv(dataset_file_path)
    if set == 'test':
        df = df.reset_index()
    return df

def yield_prediction_few_shot(
        train_dataset_file_path: str,
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
        dataset_name: Literal['Buchwald-Hartwig', 'Suzuki']
) -> None:
    
    """
    Arguments:
        - train_dataset_file_path: the csv file path of the training dataset
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - dataset_name: one of the ['Buchwald-Hartwig', 'Suzuki']
    """

    assert dataset_name in ['Buchwald-Hartwig', 'Suzuki'], f"In yield prediction, dataset name should be in ['Buchwald-Hartwig', 'Suzuki']"
    train_data = _yield_prediction_load_datasets(train_dataset_file_path, set='train')
    test_data = _yield_prediction_load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    input_col = 'reaction'
    output_col = 'high_yield'

    for sample_method, sample_num, mol_format in YIELD_PARAMS:

        prompts_log_file = result_folder + f"fs_yield_prediction_{model_name}_{mol_format}_{sample_num}_{sample_method}.log"
        performance_file = result_folder + f"fs_performance_yield_prediction_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"
        predicted_details_file = result_folder + f"fs_predicted_details_yield_prediction_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"

        if os.path.exists(performance_file):
            print(f"{performance_file} exits, continue to next params")
            continue

        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(prompts_log_file, "a") as file:
            file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

        # store
        predicted_details = []
        performance_results = []
        predicted_outputs = []
    
        # restore previous data and skip if exists
        previous_index = []
        if os.path.exists(predicted_details_file):
            previous = pd.read_csv(predicted_details_file)
            previous_index = list(previous['index'])

            predicted_details = previous.values.tolist()
            predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

        for idx, row in tqdm(test_data.iterrows()):
            input = row[input_col]
            output = row[output_col]
            index = row['index']

            if index in previous_index:
                continue

            # sample ICL examples
            if sample_method == 'Random':
                chunk = train_data.sample(sample_num, random_state=42)
            else:
                raise ValueError("Unsupported other sample method!")

            sample_examples = list(zip(chunk[input_col].values, chunk[output_col].values))

            # build prompt and save
            prompt = create_few_shot_prompt_yield_prediction(input, sample_examples, dataset_name)
            with open(prompts_log_file, "a") as file:
                file.write(prompt + "\n")
                file.write("=" * 50 + "\n")

            # model inference
            predicted_output = model(prompt)

            predicted_outputs.append(predicted_output)
            predicted_details.append([index, input] + [output] + predicted_output)

        details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
        details_df.to_csv(predicted_details_file, index=False)

        # evaluate
        acc_list = []
        # each evaluation experiment is repeated five times and the mean and variance are reported.
        repeat_times = 5
        for repeat in range(repeat_times):
            tpredicted_outputs = [i[repeat] for i in predicted_outputs]
            correct = 0
            all_num = len(test_data)
            for idx, gt in enumerate(list(test_data[output_col])):
                pred = tpredicted_outputs[idx]

                if gt.strip() == pred.strip():
                    correct += 1
            acc = correct / all_num
            acc_list.append(acc)
        
        # save results to file
        performance_results.append([model_name, mol_format, sample_num, sample_method, np.mean(acc_list)] + acc_list)

        tem = pd.DataFrame(performance_results, columns=['model_name', 'mol_format', 'sample_num', 'sample_method', 'avg_metric'] \
                        + [f'metric_{i}' for i in range(repeat_times)])
        tem.to_csv(performance_file, index=False)

def yield_prediction_zero_shot(
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
        dataset_name: Literal['Buchwald-Hartwig', 'Suzuki']
) -> None:

    """
    Arguments:
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - dataset_name: one of the ['Buchwald-Hartwig', 'Suzuki']
    """

    assert dataset_name in ['Buchwald-Hartwig', 'Suzuki'], f"In yield prediction, dataset name should be in ['Buchwald-Hartwig', 'Suzuki']"
    test_data = _yield_prediction_load_datasets(test_dataset_file_path, set='test')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    input_col = 'reaction'
    output_col = 'high_yield'

    prompts_log_file = result_folder + f"zs_yield_prediction_{model_name}.log"
    performance_file = result_folder + f"zs_performance_yield_prediction_{model_name}.csv"
    predicted_details_file = result_folder + f"zs_predicted_details_yield_prediction_{model_name}.csv"

    if os.path.exists(performance_file):
        print(f"{performance_file} exits, continue to next params")
        raise RuntimeError("exist")

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(prompts_log_file, "a") as file:
        file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

    # store
    predicted_details = []
    performance_results = []
    predicted_outputs = []

    # restore previous data and skip if exists
    previous_index = []
    if os.path.exists(predicted_details_file):
        previous = pd.read_csv(predicted_details_file)
        previous_index = list(previous['index'])

        predicted_details = previous.values.tolist()
        predicted_outputs = previous[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].values.tolist()

    for idx, row in tqdm(test_data.iterrows()):
        input = row[input_col]
        output = row[output_col]
        index = row['index']

        if index in previous_index:
            continue

        # build prompt and save
        prompt = create_zero_shot_prompt_yield_prediction(input, dataset_name)
        with open(prompts_log_file, "a") as file:
            file.write(prompt + "\n")
            file.write("=" * 50 + "\n")

        # model inference
        predicted_output = model(prompt)

        predicted_outputs.append(predicted_output)
        predicted_details.append([index, input] + [output] + predicted_output)

    details_df = pd.DataFrame(predicted_details, columns=['index', input_col, output_col, 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    details_df.to_csv(predicted_details_file, index=False)

    # evaluate
    acc_list = []
    # each evaluation experiment is repeated five times and the mean and variance are reported.
    repeat_times = 5
    for repeat in range(repeat_times):
        tpredicted_outputs = [i[repeat] for i in predicted_outputs]
        correct = 0
        all_num = len(test_data)
        for idx, gt in enumerate(list(test_data[output_col])):
            pred = tpredicted_outputs[idx]

            if gt.strip() == pred.strip():
                correct += 1
        acc = correct / all_num
        acc_list.append(acc)
    
    # save results to file
    performance_results.append([model_name, np.mean(acc_list)] + acc_list)

    tem = pd.DataFrame(performance_results, columns=['model_name', 'avg_metric'] \
                    + [f'metric_{i}' for i in range(repeat_times)])
    tem.to_csv(performance_file, index=False)
