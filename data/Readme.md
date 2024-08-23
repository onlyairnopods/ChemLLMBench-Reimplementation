The datasets are downloaded from: 
| Dataset  | Link  |  Reference | 
|  ----  | ----  |  ----  |
| USPTO_Mixed  | [download](https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144882141119) |  https://github.com/MolecularAI/Chemformer     | 
| USPTO-50k  | [download](https://az.app.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq/folder/144882141119) |  https://github.com/MolecularAI/Chemformer     |
| ChEBI-20   | [download](https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data)  |   https://github.com/blender-nlp/MolT5   |
| Suzuki-miyaura |[download](https://github.com/seokhokang/reaction_yield_nn/blob/main/data/dataset_2_0.npz)| https://github.com/seokhokang/reaction_yield_nn|
|Butchward-Hariwig|[download](https://github.com/seokhokang/reaction_yield_nn/blob/main/data/dataset_1_0.npz)|https://github.com/seokhokang/reaction_yield_nn|
| BBBP,BACE,HIV,Tox21,Clintox| [download](https://github.com/hwwang55/MolR/tree/master/data)|https://github.com/hwwang55/MolR|

# Description

*Note*: The following training dataset is the remaining part after removing the test dataset based on the original code repo from the complete dataset.

[molecule_captioning](molecule_captioning/):   
- `molecule_captioning_train.csv`, totaly 26407 entries.
- `molecule_captioning_test.csv`, totaly 100 entries.

[molecule_design](molecule_design/):   
- `molecule_design_train.csv`, totaly 26407 entries.
- `molecule_design_test.csv`, totaly 100 entries.

[name_prediction](name_prediction/):   
- `llm_test.csv`, totaly 600 entries. We randomly sample 500 molecules as the ICL candidates, and the other 100 molecules as the test set with the same `random_state=42` in [original code](https://github.com/ChemFoundationModels/ChemLLMBench/blob/main/Name_Prediction.ipynb)

[property_prediction](property_prediction/):  
1. BACE：
   - `BACE.csv`, totaly 1513 entries.
   - `BACE_train.csv`, totaly 1413 entries.
   - `BACE_test.csv`, totaly 100 entries.
2. BBBP:
   - `BBBP.csv`, totaly 2050 entries.
   - `BBBP_train.csv`, totaly 1950 entries.
   - `BBBP_test.csv`, totaly 100 entries.
3. ClinTox:  
   - `ClinTox.csv`, totaly 1484 entries.
   - `ClinTox_train.csv`, totaly 1384 entries.
   - `ClinTox_test.csv`, totaly 100 entries.
4. HIV:  
   - `HIV.csv`, totaly 41127 entries.
   - `HIV_train.csv`, totaly 41027 entries.
   - `HIV_test.csv`, totaly 100 entries.
5. Tox21:  
   - `Tox21.csv`, totaly 8014 entries.
   - `Tox_train.csv`, totaly 7914 entries.
   - `Tox_test.csv`, totaly 100 entries.

[reaction_prediction](reaction_prediction/)
- `uspto_train.csv`, totaly 409035 entries.
- `uspto_test.csv`, totaly 100 entries.

[reagent_selection](reagent_selection/)  
from [here](https://github.com/ChemFoundationModels/ChemLLMBench/tree/main/data/reagent_selection)

[retro](retro/)
- `uspto_50k_retro_train.csv`, totaly 40029 entries.
- `uspto_50k_retro_test.csv`, totaly 7914 entries.

[yield_prediction](yield_prediction/)
- `BH_dataset.csv`, totaly 3955 entries, training set.
- `BH_sample_100_test.csv`, totaly 100 entries.
- `Suzuki.csv`, totaly 5760 entries.
- `SU_train.csv`, totaly 5660 entries.
- `SU_sample_100_test.csv`, totaly 100 entries.