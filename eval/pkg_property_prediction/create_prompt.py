"""
prompts are borrowed from: https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/data/property_prediction/property_prediction_prompt.txt
"""

from typing import Union, Sequence, Literal, List, Tuple

def get_input_output_columns_by_task(task: Literal["BACE", "BBBP", "HIV", "ClinTox", "Tox21"]) -> Union[Tuple[str, str], Tuple[List[str], str]]:
    """
    get the name of the column contains input smiles, other info(if has), output label
    """
    if task == 'BACE':
        return "mol", "Class"
    elif task == 'BBBP':
        return "smiles", "p_np"
    elif task == 'HIV':
        return ["smiles", "activity"], "HIV_active"
    elif task == "ClinTox":
        return ["smiles", "FDA_APPROVED"], "CT_TOX"
    elif task == "Tox21":
        return "smiles", "SR-p53"

def create_bace_few_shot_prompt(input_smiles: str, pp_examples: List[Sequence[str]]) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the molecular properties of a given chemical compound based on its structure, by analyzing wether it can inhibit(Yes) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit(No) BACE1. Consider factors such as molecular weight, atom count, bond types, and functional groups in order to assess the compound's drug-likeness and its potential to serve as an effective therapeutic agent for Alzheimer's disease. You will be provided with task template, please answer with only Yes or No. \n"
    for example in pp_examples:
        prompt += f"SMILES: {example[0]}\nBACE-1 Inhibit: {example[-1]}\n"
    prompt += f"SMILES: {input_smiles}\nBACE-1 Inhibit:"
    return prompt

def create_bace_zero_shot_prompt(input_smiles: str) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the molecular properties of a given chemical compound based on its structure, by analyzing wether it can inhibit(Yes) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit(No) BACE1. Consider factors such as molecular weight, atom count, bond types, and functional groups in order to assess the compound's drug-likeness and its potential to serve as an effective therapeutic agent for Alzheimer's disease. You will be provided with task template, please answer with only Yes or No. \n"
    prompt += f"SMILES: {input_smiles}\nBACE-1 Inhibit:"
    return prompt

def create_bbbp_few_shot_prompt(input_smiles: str, pp_examples: List[Sequence[str]]) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. \nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically penetration/non-penetration to the brain-blood barrier, based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether it has penetrative property (Yes) or not (No). The task is to predict the binary label for a given molecule, please answer with only Yes or No.\n"
    for example in pp_examples:
        prompt += f"SMILES: {example[0]}\nPenetration: {example[-1]}\n"
    prompt += f"SMILES: {input_smiles}\nPenetration:"
    return prompt

def create_bbbp_zero_shot_prompt(input_smiles: str) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. \nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically penetration/non-penetration to the brain-blood barrier, based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether it has penetrative property (Yes) or not (No). The task is to predict the binary label for a given molecule, please answer with only Yes or No.\n"
    prompt += f"SMILES: {input_smiles}\nPenetration:"
    return prompt

def create_hiv_few_shot_prompt(input_smiles: Tuple[str, str], pp_examples: List[Sequence[str]]) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Please strictly follow the format, no other information can be provided. Given the SELFIES string of a molecule, the task focuses on predicting molecular properties, specifically inhibit of HIV replication based on the SELFIES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether a molecule can inhibit (Yes) or cannot inhibit (No) HIV replication. Additionally, the activity test results of the molecules are provided. There are three classes of the activity test: 1). CA: confirmed active, 2). CM: Confirmed moderately active 3.) CI: Confirmed inactive. The task is to precisely predict the binary label for a given molecule and its HIV activity test, considering its properties and its potential to impede HIV replication.\n"
    for example in pp_examples:
        prompt += f"SMILES: {example[0][0]}\nActivity test result: {example[0][1][0]}\nInhibit: {example[-1]}\n"
    prompt += f"SMILES: {input_smiles[0]}\nActivity test result: {input_smiles[1]}\nInhibit:"
    return prompt

def create_hiv_zero_shot_prompt(input_smiles: Tuple[str, str]) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Please strictly follow the format, no other information can be provided. Given the SELFIES string of a molecule, the task focuses on predicting molecular properties, specifically inhibit of HIV replication based on the SELFIES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether a molecule can inhibit (Yes) or cannot inhibit (No) HIV replication. Additionally, the activity test results of the molecules are provided. There are three classes of the activity test: 1). CA: confirmed active, 2). CM: Confirmed moderately active 3.) CI: Confirmed inactive. The task is to precisely predict the binary label for a given molecule and its HIV activity test, considering its properties and its potential to impede HIV replication.\n"
    prompt += f"SMILES: {input_smiles[0]}\nActivity test result: {input_smiles[1]}\nInhibit:\n"
    return prompt

def create_clintox_few_shot_prompt(input_smiles: Tuple[str, str], pp_examples: List[Sequence[str]]) -> str:
    """
    pp_examples: List[(smiles, ['Y/N'], score, 'Y/N'),]
    """
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials(Yes) or Not approved by the FDA for clinical trials(No). You will be provided with task template. The task is to predict the binary label for a given molecule, please answer with only Yes or No.\n"
    for example in pp_examples:
        prompt += f"SMILES: {example[0][0]}\nFDA-approved: {example[0][1][0]}\nClinically-trail-toxic: {example[-1]}\n"
    prompt += f"SMILES: {input_smiles[0]}\nFDA-approved: {input_smiles[1]}\nClinically-trail-toxic:"
    return prompt

def create_clintox_zero_shot_prompt(input_smiles: Tuple[str, str]) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials(Yes) or Not approved by the FDA for clinical trials(No). You will be provided with task template. The task is to predict the binary label for a given molecule, please answer with only Yes or No.\n"
    prompt += f"SMILES: {input_smiles[0]}\nFDA-approved: {input_smiles[1]}\nClinically-trail-toxic:"
    return prompt

def create_tox21_few_shot_prompt(input_smiles: str, pp_examples: List[Sequence[str]]) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is toxic(Yes) or Not toxic(No), based on the SMILES string representation of each molecule. A template will be provided. The task is to predict the binary label for a given molecule SMILES, please answer with only Yes or No.\n"
    for example in pp_examples:
        prompt += f"SMILES: {example[0]}\nPenetration: {example[-1]}\n"
    prompt += f"SMILES: {input_smiles}\nPenetration:"
    return prompt

def create_tox21_zero_shot_prompt(input_smiles: str) -> str:
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is toxic(Yes) or Not toxic(No), based on the SMILES string representation of each molecule. A template will be provided. The task is to predict the binary label for a given molecule SMILES, please answer with only Yes or No.\n"
    prompt += f"SMILES: {input_smiles}\nPenetration:"
    return prompt

def create_few_shot_prompt(input_smiles: Union[str, Sequence[str]], pp_examples: List[Tuple[str, str]], task: str) -> str:
    if task == "BACE":
        return create_bace_few_shot_prompt(input_smiles, pp_examples)
    elif task == "BBBP":
        return create_bbbp_few_shot_prompt(input_smiles, pp_examples)
    elif task == "HIV":
        return create_hiv_few_shot_prompt(input_smiles, pp_examples)
    elif task == "ClinTox":
        return create_clintox_few_shot_prompt(input_smiles, pp_examples)
    elif task == "Tox21":
        return create_tox21_few_shot_prompt(input_smiles, pp_examples)

def create_zero_shot_prompt(input_smiles: str, task: str) -> str:
    if task == "BACE":
        return create_bace_zero_shot_prompt(input_smiles)
    elif task == "BBBP":
        return create_bbbp_zero_shot_prompt(input_smiles)
    elif task == "HIV":
        return create_hiv_zero_shot_prompt(input_smiles)
    elif task == "ClinTox":
        return create_clintox_zero_shot_prompt(input_smiles)
    elif task == "Tox21":
        return create_tox21_zero_shot_prompt(input_smiles)