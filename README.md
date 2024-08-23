# Reimplementation of ChemLLMBench in Python files

This repository contains an *unofficial* reimplementation of this paper in python files: [**What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks**](https://arxiv.org/abs/2305.18365). The original code repository see [ChemLLMBench](https://github.com/ChemFoundationModels/ChemLLMBench).

# Note

1. Reimplemented the ipynb files in the original repository into Python files for scripting calls, aiming to assist in the research of large language models in the fields of chemistry and molecules.
2. Implemented some codes that was not present in the original repository (such as codes that predicted molecule properties outside of BACE task).
3. Uploaded data that was not included in the original repository. After removing the test dataset based on the original repository from the complete dataset, the remaining part was extracted as the training set.
4. <font color="red"> Due to the huge consumption of time and computation, the code has not been thoroughly tested, which means there may be potential bugs. </font> Please feel free to contact me if you have any questions.

# Files structure

1. [data](./data/) contains the training data and test data. See more [here](./data/Readme.md).
2. [eval](./eval/) contains:  
   - `mol2cap_cap2mol.py` and `pkg_mol2cap_cap2mol` contain the evaluation code for molecule captioning task and molecule design task, mainly modified from [here](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Molecule_Design.ipynb).
   - `name_prediction.py` and `pkg_name_prediction` contain the evaluation code for name prediction task, including *iupac2smiles, formula2smiles, smiles2iupac, and smiles2formula*, mainly modified from [here](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Name_Prediction.ipynb).
   - `property_prediction.py` and `pkg_property_prediction` contain the evaluation code for molecule property prediction task, including *BACE, BBBP, HIV, ClinTox, and Tox21*, mainly modified from [here](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Property_Prediction.ipynb).
   - `reaction_level_prediction.py` and `pkg_reaction_level_prediction` contain the evaluation code for four reaction level prediction tasks, including *Yield Prediction, Reaction Prediction, Reagents Selection, and Retrosynthesis*, mainly modified from [here](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Reaction_Prediction.ipynb).
   - `__init__.py` contains a function to set seed.

## More details

1. Use `Random` instead of `Fixed_iCL` in the original code to indicate the random selection of ICL samples.
2. Regarding the calculation of AUC-ROC scores in molecule property prediction task, we implemented it according to [here](https://github.com/ChemFoundationModels/ChemLLMBench/issues/8). However, simply judging whether the output of the LLMs is "Yes" or "No" and performing statistics may have potential problems in practice, as the LLMs may not be able to directly and correctly provide only "Yes" or "No", which may result in inaccurate calculation.
3. Regarding the Tox21 task in the molecule property prediction tasks, as there is no original code, it is unclear what the data input and ground-truth label are. Based on the [log file](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/data/property_prediction/test_tox21_gpt-4_8_scaffold.log), it is speculated that "SR-p53" was used as the ground-truth label.

# Usage  

<details>
<summary> Model </summary>

Define your custom model with a callable function that could accepts a string as an argument (prompt) and returns a string (response).

```python
from example_model import Model

model_name = "X-LANCE/ChemDFM-13B-v1.0" # for example
model = Model(model_name_or_id=model_name, temperature=0.2, max_new_tokens=32)
```
</details>

<details>
<summary> Name Prediction </summary>

include iupac2smiles, formula2smiles, smiles2iupac, smiles2formula

```python
from eval.name_prediction import name_prediction_few_shot, name_prediction_zero_shot

name_prediction_few_shot(
    dataset_file_path='./data/name_prediction/llm_test.csv',
    result_folder='./result/name_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model
)

name_prediction_zero_shot(
    dataset_file_path='./data/name_prediction/llm_test.csv',
    result_folder='./result/name_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model
)
```
</details>

<details>
<summary> Molecule Captioning & Molecule Design </summary>

```python
from eval.mol2cap_cap2mol import mol2cap_cap2mol_few_shot, mol2cap_cap2mol_zero_shot

# molecule captioning
mol2cap_cap2mol_few_shot(
    train_dataset_file_path='./data/molecule_captioning/molecule_captioning_train.csv',
    test_dataset_file_path='./data/molecule_captioning/molecule_captioning_test.csv',
    result_folder='./result/molecule_captioning/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='molecule_captioning'
)

mol2cap_cap2mol_zero_shot(
    test_dataset_file_path='./data/molecule_captioning/molecule_captioning_test.csv',
    result_folder='./result/molecule_captioning/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='molecule_captioning'
)

# molecule design
mol2cap_cap2mol_few_shot(
    train_dataset_file_path='./data/molecule_design/molecule_design_train.csv',
    test_dataset_file_path='./data/molecule_design/molecule_design_test.csv',
    result_folder='./result/molecule_design/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='molecule_design'
)

mol2cap_cap2mol_zero_shot(
    test_dataset_file_path='./data/molecule_design/molecule_design_test.csv',
    result_folder='./result/molecule_design/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='molecule_design'
)
```
</details>

<details>
<summary> Molecule Captioning & Molecule Design </summary>

include BACE, BBBP, HIV, ClinTox, Tox21

```python
from eval.property_prediction import property_prediction_few_shot, property_prediction_zero_shot

property_prediction_few_shot(
    train_dataset_file_path='./data/property_prediction/BACE_train.csv',
    test_dataset_file_path='./data/property_prediction/BACE_test.csv',
    result_folder='./result/property_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='BACE'
)

property_prediction_zero_shot(
    test_dataset_file_path='./data/property_prediction/HIV_test.csv',
    result_folder='./result/property_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='HIV'
)
```
</details>

<details>
<summary> Reaction-relatived Prediction </summary>

```python
from eval.reaction_level_prediction import \
    reaction_prediction_few_shot,\
    reaction_prediction_zero_shot,\
    reagents_selection,\
    retrosynthesis_few_shot,\
    retrosynthesis_zero_shot,\
    yield_prediction_few_shot,\
    yield_prediction_zero_shot

reaction_prediction_few_shot(
    train_dataset_file_path='./data/reaction_prediction/uspto_train.csv',
    test_dataset_file_path='./data/reaction_prediction/uspto_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model
)
reaction_prediction_zero_shot(
    test_dataset_file_path='./data/reaction_prediction/uspto_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model
)


reagents_selection(
    dataset_file_path='./data/reagent_selection/ligand_sample.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='ligand'
)
reagents_selection(
    dataset_file_path='./data/reagent_selection/reactant_sample.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='reactant'
)
reagents_selection(
    dataset_file_path='./data/reagent_selection/solvent_sample.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    task='solvent'
)


retrosynthesis_few_shot(
    train_dataset_file_path='./data/retro/uspto50k_retro_train.csv',
    test_dataset_file_path='./data/retro/uspto50k_retro_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model
)
retrosynthesis_zero_shot(
    test_dataset_file_path='./data/retro/uspto50k_retro_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model
)


yield_prediction_few_shot(
    train_dataset_file_path='./data/yield_prediction/BH_dataset.csv',
    test_dataset_file_path='./data/yield_prediction/BH_sample_100_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    dataset_name='Buchwald-Hartwig'
)
yield_prediction_zero_shot(
    test_dataset_file_path='./data/yield_prediction/BH_sample_100_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    dataset_name='Buchwald-Hartwig'
)

yield_prediction_few_shot(
    train_dataset_file_path='./data/yield_prediction/SU_train.csv',
    test_dataset_file_path='./data/yield_prediction/SU_sample_100_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    dataset_name='Suzuki'
)
yield_prediction_zero_shot(
    test_dataset_file_path='./data/yield_prediction/SU_sample_100_test.csv',
    result_folder='./result/reaction_prediction/',
    model_name='ChemDFM-13B-v1.0',
    model=model,
    dataset_name='Suzuki'
)
```
</details>

# Citation
Cite the original paper:

```bibtex
@misc{guo2023gpt,
      title={What indeed can GPT models do in chemistry? A comprehensive benchmark on eight tasks}, 
      author={Taicheng Guo and Kehan Guo and Bozhao Nan and Zhenwen Liang and Zhichun Guo and Nitesh V. Chawla and Olaf Wiest and Xiangliang Zhang},
      year={2023},
      eprint={2305.18365},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you find this repository helpful, please consider citing it as follows:

```bibtex
@misc{Lai_Reimplementation_of_ChemLLMBench,
    title = {Reimplementation of "What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks"},
    author = {Zhengzhao Lai},
    url = {https://github.com/onlyairnopods/ChemLLMBench-Reimplementation},
    year = {2024}
}
```