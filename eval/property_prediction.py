import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import datetime
import random
import statistics
from typing import List, Any, Literal, Optional, Union, Tuple, Sequence, Callable
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.debug')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.info')

from .pkg_property_prediction.sample import top_k_scaffold_similar_molecules, random_sample_examples
from .pkg_property_prediction.create_prompt import create_few_shot_prompt, create_zero_shot_prompt, get_input_output_columns_by_task
from sklearn.metrics import accuracy_score, f1_score

random.seed(42)

SAMPLE_NUMS = [4, 8]
SAMPLE_METHODS = ['Random', 'Scaffold_SIM']

def _load_datasets(dataset_file_path: str, task: str, input_col: Union[str, List[str]], output_col: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_file_path)
    if task in ["BACE", "BBBP", "HIV", "Tox21"]:
        df[output_col] = df[output_col].apply(lambda x: "Yes" if x == 1 else "No")
    elif task in ["ClinTox"]:
        df[input_col[1]] = df[input_col[1]].apply(lambda x: "Yes" if x == 1 else "No")
        df[output_col] = df[output_col].apply(lambda x: "Yes" if x == 1 else "No")

    return df

def property_prediction_few_shot(
        train_dataset_file_path: str,
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
        task: Literal["BACE", "BBBP", "HIV", "ClinTox", "Tox21"]
) -> None:
    
    """
    Arguments:
        - train_dataset_file_path: the csv file path of the training dataset
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - task: one of the ["BACE", "BBBP", "HIV", "ClinTox", "Tox21"]
    """

    input_col, output_col = get_input_output_columns_by_task(task)
    train_data = _load_datasets(train_dataset_file_path, task, input_col, output_col)
    test_data = _load_datasets(test_dataset_file_path, task, input_col, output_col)

    paras = 0
    for sample_method in SAMPLE_METHODS:
        for sample_num in SAMPLE_NUMS:

            if paras < 0:
                paras += 1
                continue
            
            prompts_log_file = result_folder + f"fs_{task}_{model_name}_{sample_num}_{sample_method}.log"
            performance_file = result_folder + f"fs_performance_{task}_{model_name}_{sample_num}_{sample_method}.csv"
            predicted_details_file = result_folder + f"fs_predicted_details_{task}_{model_name}_{sample_num}_{sample_method}.csv"

            if os.path.exists(predicted_details_file):
                predicted_details = pd.read_csv(predicted_details_file)
                #convert the column to list
                predicted_details = predicted_details.values.tolist()
            else:
                predicted_details = []

            now = datetime.datetime.now()
            date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(prompts_log_file, "a") as file:
                file.write("=" * 30 + date_time_str + "=" * 30 + "\n")

            performance_results = []
            predicted_outputs = []

            if sample_method == 'Random':
                sample_examples = random_sample_examples(train_data, sample_num, input_col, output_col)

            for i in tqdm(range(0, len(test_data))):
                if isinstance(input_col, str):
                    example = [(test_data.iloc[i][input_col],test_data.iloc[i][output_col])]
                elif isinstance(input_col, list):
                    example = [(test_data.iloc[i][input_col].values.tolist(),test_data.iloc[i][output_col])]

                for text in example:
                    if sample_method == 'Scaffold_SIM':
                        sample_examples = top_k_scaffold_similar_molecules(text[0], train_data, input_col, output_col, sample_num)
                    prompt = create_few_shot_prompt(text[0], sample_examples, task)
                    with open(prompts_log_file, "a") as file:
                        file.write(prompt + "\n")
                        file.write("=" * 50 + "\n")

                    # model inference
                    generated_p = model(prompt)

                    # generated_p = [1 if i == "Yes" else 0 for i in generated_p]
                    predicted_outputs.append(generated_p)
                    predicted_details.append([i for i in text[0] if isinstance(text[0], list)] + ([text[0]] if isinstance(text[0], str) else []) + [text[-1]] + generated_p)

            details_df = pd.DataFrame(predicted_details, columns=[f'{task}_smiles'] + [str(x) for x in input_col[1:] if isinstance(input_col, list)] + ['class_label', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
            details_df.to_csv(predicted_details_file, index=False)

            # evaluate
            acc_list = []
            f1_list = []
            pred_list = []
            gt_list = []
            # each evaluation experiment is repeated five times and the mean and variance are reported.
            repeat_times = 5
            for repeat in range(repeat_times):
                tpredicted_outputs = [i[repeat] for i in predicted_outputs]
                for idx, gt in enumerate(list(test_data[output_col])):
                    gt = 1 if gt == "Yes" else 0
                    pred = tpredicted_outputs[idx].strip()

                    if pred == "Yes":
                        _pred = 1
                    elif pred == "No":
                        _pred = 0
                    else:
                        print(f"Got unexpected prediction: {pred} !")
                        continue
                
                    gt_list.append(int(gt))
                    pred_list.append(int(_pred))

                acc = accuracy_score(gt_list, pred_list)
                f1 = f1_score(gt_list, pred_list)
                acc_list.append(acc)
                f1_list.append(f1)

            performance_results.append([model_name] + [np.mean(acc_list)] + acc_list + [np.mean(f1_list)] + f1_list)
            tem = pd.DataFrame(performance_results, columns=['model_name', 'avg_acc'] + [f'acc_{i}' for i in range(repeat_times)] + ['avg_f1'] + [f'f1_{i}' for i in range(repeat_times)])
            tem.to_csv(performance_file, index=False)
            print(f"paras: {paras}\n")

def property_prediction_zero_shot(
        test_dataset_file_path: str,
        result_folder: str, 
        model_name: str,
        model: Callable[[str], str],
        task: Literal["BACE", "BBBP", "HIV", "ClinTox", "Tox21"]
) -> None:
    
    """
    Arguments:
        - test_dataset_file_path: the csv file path of the test dataset
        - result_folder: the folder to save the results
        - model_name: the name of the model
        - model: the model to be tested. Should be callable.
        - task: one of the ["BACE", "BBBP", "HIV", "ClinTox", "Tox21"]
    """

    input_col, output_col = get_input_output_columns_by_task(task)
    test_data = _load_datasets(test_dataset_file_path, task, input_col, output_col)

    prompts_log_file = result_folder + f"zs_{task}_{model_name}.log"
    performance_file = result_folder + f"zs_performance_{model_name}.csv"
    predicted_details_file = result_folder + f"zs_predicted_details_{task}_{model_name}.csv"

    if os.path.exists(predicted_details_file):
        predicted_details = pd.read_csv(predicted_details_file)
        #convert the column to list
        predicted_details = predicted_details.values.tolist()
    else:
        predicted_details = []

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(prompts_log_file, "a") as file:
        file.write("=" * 30 + date_time_str + "=" * 30 + "\n")
    
    performance_results = []
    predicted_outputs = []

    for i in tqdm(range(0, len(test_data))):
        if isinstance(input_col, str):
            example = [(test_data.iloc[i][input_col],test_data.iloc[i][output_col])]
        elif isinstance(input_col, list):
            example = [(test_data.iloc[i][input_col].values.tolist(),test_data.iloc[i][output_col])]

        for text in example:
            prompt = create_zero_shot_prompt(text[0], task)
            with open(prompts_log_file, "a") as file:
                file.write(prompt + "\n")
                file.write("=" * 50 + "\n")

            # model inference
            generated_p = model(prompt)

            # generated_p = [1 if i == "Yes" else 0 for i in generated_p]
            predicted_outputs.append(generated_p)
            predicted_details.append([i for i in text[0] if isinstance(text[0], list)] + ([text[0]] if isinstance(text[0], str) else []) + [text[-1]] + generated_p)

    details_df = pd.DataFrame(predicted_details, columns=[f'{task}_smiles'] + [str(x) for x in input_col[1:] if isinstance(input_col, list)] + ['class_label', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    details_df.to_csv(predicted_details_file, index=False)

    # evaluate
    acc_list = []
    f1_list = []
    pred_list = []
    gt_list = []
    # each evaluation experiment is repeated five times and the mean and variance are reported.
    repeat_times = 5
    for repeat in range(repeat_times):
        tpredicted_outputs = [i[repeat] for i in predicted_outputs]
        for idx, gt in enumerate(list(test_data[output_col])):
            gt = 1 if gt == "Yes" else 0
            pred = tpredicted_outputs[idx].strip()

            if pred == "Yes":
                _pred = 1
            elif pred == "No":
                _pred = 0
            else:
                print(f"Got unexpected prediction: {pred} !")
                continue
        
            gt_list.append(int(gt))
            pred_list.append(int(_pred))

        acc = accuracy_score(gt_list, pred_list)
        f1 = f1_score(gt_list, pred_list)
        acc_list.append(acc)
        f1_list.append(f1)
    
    performance_results.append([model_name] + [np.mean(acc_list)] + acc_list + [np.mean(f1_list)] + f1_list)
    tem = pd.DataFrame(performance_results, columns=['model_name', 'avg_acc'] + [f'acc_{i}' for i in range(repeat_times)] + ['avg_f1'] + [f'f1_{i}' for i in range(repeat_times)])
    tem.to_csv(performance_file, index=False)
