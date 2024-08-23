from typing import Sequence, List, Tuple, Literal

def create_prompt_molecule_design(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES \
string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable. \n"
    
    for example in examples:
        prompt += f"Molecular requirements description: {example[0]}\nMolecular SMILES: {example[1]}\n"
    prompt += f"Molecular requirements description: {input_text}\nMolecular SMILES:"
    return prompt

def create_prompt_molecule_captioning(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"
    
    for example in examples:
        prompt += f"Molecule SMILES: {example[0]}\nMolecular Description: {example[1]}\n"
    prompt += f"Molecule SMILES: {input_text}\nMolecular Description:"
    return prompt

def create_few_shot_prompt(reactant: str, examples: List[Tuple[str, str]], task: Literal['molecule_design', 'molecule_captioning']) -> str:
    if task == 'molecule_design':
        return create_prompt_molecule_design(reactant, examples)
    elif task == 'molecule_captioning':
        return create_prompt_molecule_captioning(reactant, examples)
   
def create_zero_shot_prompt(input_text: str, task: Literal['molecule_design', 'molecule_captioning']):
    if task == 'molecule_design':
        prompt = "You are an expert chemist. Given the molecular requirements description: '{}', design the molecular SMILES using your experienced chemical molecule design knowledge. No explanations and other information. \
Only return the designed molecular SMILES. The SMILES must be valid and chemically reasonable.".format(input_text)
        
    elif task == 'molecule_captioning':
        prompt = "You are an expert chemist. Given the molecular SMILES: {}, provide the detailed molecule description using your experienced chemical Molecular knowledge.".format(input_text)
    
    return prompt

def get_input_output_columns_by_task(task: Literal['molecule_design', 'molecule_captioning']) -> Tuple[str, str]:
    if task == 'molecule_design':
        return "description", "SMILES"
    elif task == 'molecule_captioning':
        return "SMILES", "description"