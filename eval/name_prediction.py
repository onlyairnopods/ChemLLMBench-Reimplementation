import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import datetime
import statistics
from typing import List, Any, Literal, Optional, Union, Tuple, Sequence, Callable
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.debug')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

from .utils import top_n_scaffold_similar_molecules, top_n_similar_strings, get_scaffold_fp, canonicalize_smiles
from .pkg_name_prediction.create_prompt import get_input_output_columns_by_task, create_few_shot_prompt, create_zero_shot_prompt

PARAMS = [
    # sample_method, sample_num, mol_format
    ('Scaffold_SIM', 20, 'SMILES'),
    ('Scaffold_SIM', 5, 'SMILES'),
    ('Random', 20, 'SMILES'),
]

TASK_TYPE = ['iupac2smiles', 'formula2smiles', 'smiles2iupac', 'smiles2formula']

def _load_datasets(dataset_file_path: str = './llm_test.csv') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dataset_file_path)
    df = df[~df['iupac'].isna()]

    train, test = train_test_split(df, test_size=100, random_state=42)

    # len(train): 500
    # len(test): 100

    train['scaffold_fp'] = train['smiles'].apply(lambda x: get_scaffold_fp(x))
    train = train[~train['scaffold_fp'].isna()]
    train['smiles'] = train['smiles'].apply(lambda x: canonicalize_smiles(x))
    test['smiles'] = test['smiles'].apply(lambda x: canonicalize_smiles(x))
    
    test = test.reset_index()
    test = test.head(100)

    return train, test

def _few_shot_merge_performance(
        result_folder: str,
        model_name: str,
        repeat_times: int,
) -> None:
    all_performance = pd.DataFrame()
    for task in TASK_TYPE:
        for sample_method, sample_num, mol_format in PARAMS:
            performance_file = result_folder + f"few_shot_test_performance_{task}_{model_name}_{mol_format}_{sample_num}_{sample_method}.csv"
            tem = pd.read_csv(performance_file)
            all_performance = all_performance.append(tem)

    cols = ['metric_{}'.format(i) for i in range(repeat_times)]
    all_performance['std'] = all_performance[cols].apply(lambda row: statistics.stdev(row), axis=1)
    all_performance['metric'] = all_performance.apply(lambda row: "{:.3f} $\pm$ {:.3f}".format(row['avg_metric'], row['std']), axis=1)
    all_performance_file = result_folder + f"few_shot_test_all_performance_{model_name}.csv"
    all_performance.to_csv(all_performance_file, index=False)
    print(f"{'='*30} Evaluation results saved to {all_performance_file} {'='*30}")

def _zero_shot_merge_performance(
        result_folder: str,
        model_name: str,
        repeat_times: int,
        mol_format: str,
) -> None:
    all_performance = pd.DataFrame()
    for task in TASK_TYPE:
        performance_file = result_folder + f"zero_shot_test_performance_{task}_{model_name}_{mol_format}.csv"
        tem = pd.read_csv(performance_file)
        all_performance = all_performance.append(tem)

    cols = ['metric_{}'.format(i) for i in range(repeat_times)]
    all_performance['std'] = all_performance[cols].apply(lambda row: statistics.stdev(row), axis=1)
    all_performance['metric'] = all_performance.apply(lambda row: "{:.3f} $\pm$ {:.3f}".format(row['avg_metric'], row['std']), axis=1)
    all_performance_file = result_folder + f"zero_shot_test_all_performance_{model_name}.csv"
    all_performance.to_csv(all_performance_file, index=False)
    print(f"{'='*30} Evaluation results saved to {all_performance_file} {'='*30}")


def name_prediction_few_shot(
        dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
) -> None:
    
    """
    Arguments:
        - dataset_file_path: the csv file path of the dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
    """

    train_data, test_data = _load_datasets(dataset_file_path)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)

    for task in TASK_TYPE:
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
            performance_results = []
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
                    if input_col == 'smiles':
                        top_smiles = top_n_scaffold_similar_molecules(input, list(train_data['scaffold_fp']), list(train_data['smiles']), top_n=sample_num)
                    else:
                        top_smiles = top_n_similar_strings(input, list(train_data[input_col]), top_n=sample_num)
                    chunk = train_data[train_data[input_col].isin(top_smiles)]

                sample_examples = list(zip(chunk[input_col].values, chunk[output_col].values))

                # build prompt and save
                prompt = create_few_shot_prompt(input, sample_examples, task)
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
                for idx, gt in enumerate(list(test_data[output_col])[:4]):
                    pred = tpredicted_outputs[idx]
                    if task in ['iupac2smiles', 'formula2smiles']:
                        try:
                            mol = Chem.MolFromSmiles(pred)
                            pred = Chem.MolToSmiles(mol)
                        except Exception as e:
                            continue

                    if gt == pred:
                        correct += 1
                acc = correct / all_num
                acc_list.append(acc)

            # save results to file
            performance_results.append([task, model_name, mol_format, sample_num, sample_method, np.mean(acc_list)] + acc_list)

            tem = pd.DataFrame(performance_results, columns=['task', 'model_name', 'mol_format', 'sample_num', 'sample_method', 'avg_metric'] \
                            + [f'metric_{i}' for i in range(repeat_times)])
            tem.to_csv(performance_file, index=False)

    _few_shot_merge_performance(result_folder, model_name, repeat_times)

        

def name_prediction_zero_shot(
        dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
) -> None:
    
    """
    Arguments:
        - dataset_file_path: the csv file path of the dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
    """

    train_data, test_data = _load_datasets(dataset_file_path)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    mol_format = "SMILES"
    for task in TASK_TYPE:

        prompts_log_file = result_folder + f"zs_{task}_{model_name}_{mol_format}.log"
        performance_file = result_folder + f"zs_performance_{task}_{model_name}_{mol_format}.csv"
        predicted_details_file = result_folder + f"zs_predicted_details_{task}_{model_name}_{mol_format}.csv"

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
        performance_results = []
    
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
                if task in ['iupac2smiles', 'formula2smiles']:
                    try:
                        mol = Chem.MolFromSmiles(pred)
                        pred = Chem.MolToSmiles(mol)
                    except Exception as e:
                        continue

                if gt == pred:
                    correct += 1
            acc = correct / all_num
            acc_list.append(acc)

        # save results to file
        performance_results.append([task, model_name, mol_format, np.mean(acc_list) + acc_list])

        tem = pd.DataFrame(performance_results, columns=['task', 'model_name', 'mol_format', 'avg_metric'] \
                        + [f'metric_{i}' for i in range(repeat_times)])
        tem.to_csv(performance_file, index=False)

    _zero_shot_merge_performance(result_folder, model_name, repeat_times, mol_format)