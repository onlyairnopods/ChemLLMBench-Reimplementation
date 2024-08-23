from typing import Sequence, List, Tuple, Literal

def create_prompt_smiles2iupac(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the molecular SMILES, your task is to predict the IUPAC name using your experienced chemical IUPAC name knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"
    for example in examples:
        prompt += f"Molecular SMILES: {example[0]}\nMolecular IUPAC name: {example[1]}\n"
    prompt += f"Molecular SMILES: {input_text}\nMolecular IUPAC name:"
    return prompt

def create_prompt_iupac2smiles(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the molecular IUPAC name, your task is to predict the molecular SMILES using your experienced chemical IUPAC name knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with molecular SMILES string notations to represent the IUPAC name. The SMILES must be valid and chemically reasonable. \n"
    for example in examples:
        prompt += f"Molecular IUPAC name: {example[0]}\nMolecular SMILES: {example[1]}\n"
    prompt += f"Molecular IUPAC name: {input_text}\nMolecular SMILES:"
    return prompt

def create_prompt_smiles2formula(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the molecular SMILES, your task is to predict the molecular formula using your experienced chemical molecular formula knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"
    for example in examples:
        prompt += f"Molecular SMILES: {example[0]}\nMolecular formula: {example[1]}\n"
    prompt += f"Molecular SMILES: {input_text}\nMolecular formula:"
    return prompt

def create_prompt_formula2smiles(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the molecular formula, your task is to predict the molecular SMILES using your experienced chemical molecular formula knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with molecular SMILES string notations to represent the molecular formula. The SMILES must be valid and chemically reasonable.\n"
    for example in examples:
        prompt += f"Molecular formula: {example[0]}\nMolecular SMILES: {example[1]}\n"
    prompt += f"Molecular formula: {input_text}\nMolecular SMILES:"
    return prompt

def create_few_shot_prompt(reactant: str, examples: List[Tuple[str, str]], task: Literal['smiles2iupac', 'smiles2formula', 'iupac2smiles', 'formula2smiles']) -> str:
    if task == 'smiles2iupac':
        return create_prompt_smiles2iupac(reactant, examples)
    elif task == 'smiles2formula':
        return create_prompt_smiles2formula(reactant, examples)
    elif task == 'iupac2smiles':
        return create_prompt_iupac2smiles(reactant, examples)
    elif task == 'formula2smiles':
        return create_prompt_formula2smiles(reactant, examples)
    

def create_zero_shot_prompt(input_text: str, task: Literal['smiles2iupac', 'smiles2formula', 'iupac2smiles', 'formula2smiles']):
    if task == 'smiles2iupac':
        
        prompt = "You are an expert chemist. Given the molecular SMILES: {}, predict the molecular IUPAC name using your experienced chemical molecular SMILES and IUPAC name knowledge. No explanations and other information. \
Only return the molecular IUPAC name.".format(input_text)
        
    elif task == 'smiles2formula':
        
        prompt = "You are an expert chemist. Given the molecular SMILES: {}, predict the chemical molecular formula using your experienced chemical molecular SMILES and formula knowledge. No explanations and other information. \
Only return the molecular formula.".format(input_text)
        
    elif task == 'iupac2smiles':
        
        prompt = "You are an expert chemist. Given the molecular IUPAC name: {}, predict the molecular SMILES using your experienced chemical molecular IUPAC name and SMILES knowledge. No explanations and other information. \
Only return the molecular SMILES.".format(input_text)
        
    elif task == 'formula2smiles':
        
        prompt = "You are an expert chemist. Given the molecular formula: {}, predict the molecular SMILES using your experienced chemical molecular formula and SMILES knowledge. No explanations and other information. \
Only return the molecular SMILES.".format(input_text)
        
    return prompt

def get_input_output_columns_by_task(task: Literal['smiles2iupac', 'smiles2formula', 'iupac2smiles', 'formula2smiles']) -> Tuple[str, str]:
    if task == 'smiles2iupac':
        return "smiles", "iupac"
    elif task == 'smiles2formula':
        return "smiles", "formula"
    elif task == 'iupac2smiles':
        return "iupac", "smiles"
    elif task == 'formula2smiles':
        return "formula", 'smiles'