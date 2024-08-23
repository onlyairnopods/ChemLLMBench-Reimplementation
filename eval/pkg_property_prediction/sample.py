from typing import List, Any, Literal, Optional, Union, Tuple, Sequence

import pandas as pd
import warnings

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit import RDLogger

# random sampling
def random_sample_examples(data: pd.DataFrame, sample_size: int, input_col: Union[str, List[str]], output_col: str) -> List[Tuple[str, str]]:
    positive_examples = data[data[output_col] == "Yes"].sample(int(sample_size/2))
    negative_examples = data[data[output_col] == "No"].sample(int(sample_size/2))
    if isinstance(input_col, str):
        smiles = positive_examples[input_col].tolist() + negative_examples[input_col].tolist()
    elif isinstance(input_col, list):
        smiles = positive_examples[input_col].values.tolist() + negative_examples[input_col].values.tolist()
    
    class_label = positive_examples[output_col].tolist() + negative_examples[output_col].tolist()
    #convert 1 to "Yes" and 0 to "No"" in class_label
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    examples = list(zip(smiles, class_label))
    return examples

# scaffold sampling
def top_k_scaffold_similar_molecules(target_smiles: str, train_data: pd.DataFrame, input_col: str, output_col: str, top_n: int = 5) -> List[Tuple[str, str]]:
    #drop the target_smiles from the dataset
    if isinstance(input_col, list):
        other_info_list = train_data[input_col[1:]].values.tolist()
        target_smiles = target_smiles[0]
        input_col = input_col[0]

    elif isinstance(input_col, str):
        other_info_list = None
    
    molecule_smiles_list = train_data[input_col].tolist()
    train_data = train_data[train_data[input_col] != target_smiles]
    label_list = train_data[output_col].tolist()
    # label_list = ["Yes" if i == 1 else "No" for i in label_list]

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is not None:
        target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    else:
        print("Error: Unable to create a molecule from the provided SMILES string.")
        #drop the target_smiles from the dataset
        return None

    target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    target_fp = rdMolDescriptors.GetMorganFingerprint(target_scaffold, 2)
    RDLogger.DisableLog('rdApp.warning')
    warnings.filterwarnings("ignore", category=UserWarning)
    similarities = []
    
    for i, smiles in enumerate(molecule_smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            if other_info_list:
                similarities.append(((smiles, other_info_list[i]), tanimoto_similarity, label_list[i]))
            else:
                similarities.append((smiles, tanimoto_similarity, label_list[i]))
        except:
            continue
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_similar_molecules = similarities[:top_n]
    return top_n_similar_molecules