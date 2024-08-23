from typing import List, Any, Literal, Optional, Union, Tuple, Sequence

from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from fcd import get_fcd, load_ref_model, canonical_smiles
import numpy as np
import pandas as pd
import statistics
from transformers import BertTokenizerFast
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

def evaluate1(outputs: List[Tuple[str, str, str]], verbose: bool = False) -> Tuple[float, float, float, float]:
    """
    Return:
        bleu_score, exact_match_score, levenshtein_score, validity_score
    """
    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):
        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')

        gt_tokens = [c for c in gt]
        out_tokens = [c for c in out]
        references.append([gt_tokens])
        hypotheses.append(out_tokens)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose: 
        print('BLEU score:', bleu_score)

    references = []
    hypotheses = []
    levs = []

    num_exact = 0
    bad_mols = 0

    for i, (smi, gt, out) in enumerate(outputs):
        hypotheses.append(out)
        references.append(gt)
        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): 
                num_exact += 1
        except:
            bad_mols += 1

        levs.append(lev(out, gt))

    # Exact matching score
    exact_match_score = num_exact/(i+1)
    if verbose:
        print('Exact Match:', exact_match_score)

    # Levenshtein score
    levenshtein_score = np.mean(levs)
    if verbose:
        print('Levenshtein:', levenshtein_score)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)

    return bleu_score, exact_match_score, levenshtein_score, validity_score

def evaluate2(raw_outputs: List[Tuple[str, str, str]], morgan_r: int = 2, verbose: bool = False) -> Tuple[float, float, float, float]:
    """
    Return:
        validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score
    """
    bad_mols = 0
    outputs = []

    for desc, gt_smi, ot_smi in raw_outputs:
        try:
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m == None: 
                raise ValueError('Bad SMILES')
            outputs.append((desc, gt_m, ot_m))
        except:
            bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    if verbose:
        print('validity:', validity_score)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):
        if i % 100 == 0:
            if verbose: print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score

def evaluate3(gt_smis: str, ot_smis: str, verbose: bool = False) -> float:
    """
    Return:
        fcd_sim_score
    """
    model = load_ref_model()

    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]

    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    if verbose:
        print('FCD Similarity:', fcd_sim_score)

    return fcd_sim_score

def evaluate_one_molecule_design(details_df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Return:
        - [bleu_mean, exact_match_mean, levenshtein_mean, validity_mean, maccs_sims_mean, rdk_sims_mean, morgan_sims_mean, fcd_mean]
        - [bleu_variance, exact_match_variance, levenshtein_variance, validity_variance, maccs_sims_variance, rdk_sims_variance, morgan_sims_variance, fcd_variance]
    """
    bleu_scores = []
    exact_match_scores = []
    levenshtein_scores = []
    validity_scores = []
    maccs_sims_scores = []
    rdk_sims_scores = []
    morgan_sims_scores = []
    fcd_scores = []
    
    for i in range(1, 6):
        tem = list(zip(details_df['description'], details_df['SMILES'], details_df['pred_{}'.format(i)]))
        
        bleu_score, exact_match_score, levenshtein_score, _ = evaluate1(tem)
        validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = evaluate2(tem)
        fcd = evaluate3(list(details_df['SMILES']), list(details_df['pred_{}'.format(i)]))
        
        bleu_scores.append(bleu_score)
        exact_match_scores.append(exact_match_score)
        levenshtein_scores.append(levenshtein_score)
        
        validity_scores.append(validity_score)
        maccs_sims_scores.append(maccs_sims_score)
        rdk_sims_scores.append(rdk_sims_score)
        morgan_sims_scores.append(morgan_sims_score)
        
        fcd_scores.append(fcd)
    
    bleu_mean = np.mean(bleu_scores)
    exact_match_mean = np.mean(exact_match_scores)
    levenshtein_mean = np.mean(levenshtein_scores)
    validity_mean = np.mean(validity_scores)
    maccs_sims_mean = np.mean(maccs_sims_scores)
    rdk_sims_mean = np.mean(rdk_sims_scores)
    morgan_sims_mean = np.mean(morgan_sims_scores)
    fcd_mean = np.mean(fcd_scores)
    

    # cal std
    bleu_variance = statistics.stdev(bleu_scores)
    exact_match_variance = statistics.stdev(exact_match_scores) 
    levenshtein_variance = statistics.stdev(levenshtein_scores)
    validity_variance = statistics.stdev(validity_scores)
    maccs_sims_variance = statistics.stdev(maccs_sims_scores) 
    rdk_sims_variance = statistics.stdev(rdk_sims_scores) 
    morgan_sims_variance = statistics.stdev(morgan_sims_scores) 
    fcd_variance = statistics.stdev(fcd_scores) 
    
    stds = [bleu_variance, exact_match_variance, levenshtein_variance, 
             validity_variance, maccs_sims_variance, rdk_sims_variance, 
             morgan_sims_variance, fcd_variance]
    
    return [bleu_mean, exact_match_mean, levenshtein_mean, validity_mean, maccs_sims_mean, rdk_sims_mean, morgan_sims_mean, fcd_mean], stds 


def evaluate4(raw_outputs: List[Tuple[str, str, str]], text_model: str = 'allenai/scibert_scivocab_uncased', text_trunc_length : int = 512) -> Tuple[float, float, float, float, float, float]:
    """
    Return:
        bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score
    """
    outputs = []
    
    for smiles, gt, output in raw_outputs:
        out_tmp = output[6:] if output.startswith('[CLS] ') else output
        outputs.append((smiles, gt, out_tmp))

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
    meteor_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):
        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    _meteor_score = np.mean(meteor_scores)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score

def evaluate_one_molecule_captioning(details_df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Return:
        - [mean_bleu2, mean_bleu4, mean_rouge1, mean_rouge2, mean_rougel, mean_meteor]
        - [std_bleu2, std_bleu4, std_rouge1, std_rouge2, std_rougel, std_meteor]
    """
    bleu2_scores = []
    bleu4_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    meteor_scores = []
    
    for i in range(1, 6):
        tem = list(zip(details_df['SMILES'], details_df['description'], details_df['pred_{}'.format(i)]))
        bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score = evaluate4(tem)
        
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)
        rouge1_scores.append(rouge_1)
        rouge2_scores.append(rouge_2)
        rougel_scores.append(rouge_l)
        meteor_scores.append(_meteor_score)
    
    mean_bleu2 = np.mean(bleu2_scores)
    mean_bleu4 = np.mean(bleu4_scores)
    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougel = np.mean(rougel_scores)
    mean_meteor = np.mean(meteor_scores)
    
    std_bleu2 = statistics.stdev(bleu2_scores)
    std_bleu4 = statistics.stdev(bleu4_scores)
    std_rouge1 = statistics.stdev(rouge1_scores)
    std_rouge2 = statistics.stdev(rouge2_scores)
    std_rougel = statistics.stdev(rougel_scores)
    std_meteor = statistics.stdev(meteor_scores)
    
    stds = [std_bleu2, std_bleu4, std_rouge1, std_rouge2, std_rougel, std_meteor]
    return [mean_bleu2, mean_bleu4, mean_rouge1, mean_rouge2, mean_rougel, mean_meteor], stds