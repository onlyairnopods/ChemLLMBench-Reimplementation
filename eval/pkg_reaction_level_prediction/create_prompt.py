from typing import Sequence, List, Tuple, Literal

def create_few_shot_prompt_reaction_prediction(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the reactants SMILES, your task is to predict the main product SMILES using your experienced chemical Reaction Prediction knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the product. The input contains the reactants and reagents which are split by '.'. The product smiles must be valid and chemically reasonable. \n"
    
    for example in examples:
        prompt += f"Reactants+Reagents: {example[0]}\nProducts: {example[1]}\n"
    prompt += f"Reactants+Reagents: {input_text}\nProducts:"
    return prompt

def create_zero_shot_prompt_reaction_prediction(input_text: str) -> str:
    prompt = "You are an expert chemist. Given the reactants SMILES: {}, predict the reaction product SMILES using your experienced chemical Reaction Prediction knowledge. No explanations and other information. \
Only return the product SMILES.".format(input_text)
    return prompt

def create_few_shot_prompt_retrosynthesis(input_text: str, examples: List[Tuple[str, str]]) -> str:
    prompt = "You are an expert chemist. Given the product SMILES, your task is to predict the reactants SMILES using your experienced chemical Retrosynthesis knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the reactants. The output contains the reactants and reagents which are split by '.'. The reactants smiles must be valid and chemically reasonable. \n"
    
    for example in examples:
        prompt += f"Reaction Product: {example[0]}\nReactants: {example[1]}\n"
    prompt += f"Reaction Product: {input_text}\nReactants:"
    return prompt
 
def create_zero_shot_prompt_retrosynthesis(input_text: str) -> str:
    prompt = "You are an expert chemist. Given the product SMILES: {}, predict the reactants SMILES using your experienced chemical Retrosynthesis knowledge. No explanations and other information. \
You should only reply with SMILES string notations to represent the reactants. The output contains the reactants and reagents which are split by '.'. The reactants smiles must be valid and chemically reasonable.".format(input_text)
    return prompt

def create_few_shot_prompt_yield_prediction(input_text: str, examples: List[Tuple[str, str]], dataset_name: Literal['Buchwald-Hartwig', 'Suzuki']) -> str:
    prompt = f"You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical Yield Prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SMILES string of a {dataset_name} reaction, the task focuses on predicting reaction yield, specifically whether a reaction is High-yielding (Yes) or Not High-yielding (No), based on the SMILES string representation of each {dataset_name} reaction. The reactants are separated by '.', which are separated from the product by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. You will be provided with several examples molecules. Please answer with only Yes or No.\n"
    
    for example in examples:
        prompt += f"Reaction: {example[0]}\High-yielding reaction: {example[1]}\n"
    prompt += f"Reaction: {input_text}\High-yielding reaction:"
    return prompt

def create_zero_shot_prompt_yield_prediction(input_text: str, dataset_name: Literal['Buchwald-Hartwig', 'Suzuki']) -> str:
    _prompt = f"You are an expert chemist, your task is to predict the yield of reaction using your experienced chemical Yield Prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the {dataset_name} reaction SMILES: {input_text}, the task focuses on predicting reaction yield, specifically whether a reaction is High-yielding (Yes) or Not High-yielding (No), based on the SMILES string representation of each {dataset_name} reaction. The reactants are separated by '.', which are separated from the product by '>>'. High-yielding reaction means the yield rate of the reaction is above 70. Please answer with only Yes or No.\n"

    prompt = f"You are an expert chemist. Given the {dataset_name} reaction SMILES: {input_text}, predict whether the reaction is High-yielding (Yes) or Not High-yielding (No) using your experienced chemical Yield Prediction knowledge. No explanations and other information. \
Only return Yes or No."

    return prompt