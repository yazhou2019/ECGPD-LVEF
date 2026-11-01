import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# load data
from ECG_preprocess_LVEF import (y_train, y_val, y_test, X_train, X_val, X_test,
                                 label_names as shd)

import sklearn.metrics
import random

use_index = 0 # corresponds to low LVEF


y_train_j = y_train[:, use_index] if y_train.shape[1] > 1 else y_train
y_val_j = y_val[:, use_index] if y_val.shape[1] > 1 else y_val
y_test_j = y_test[:, use_index] if y_test.shape[1] > 1 else y_test

print(f"\nuse_index={use_index}, corresponding to {shd[use_index]}")
print(f"\ndata size:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")
margin_index2 = 7

X_train = X_train[:,margin_index2:]
X_val = X_val[:,margin_index2:]
X_test = X_test[:,margin_index2:]


################
# evaluation
################
import pandas as pd

df = pd.read_csv('./data/all_test.csv')
filenames_list = df['FILE_NAME'].tolist()
filenames_dic = {}
for tmpt in range(len(filenames_list)):
    filenames_dic[filenames_list[tmpt].split('.')[0]] = tmpt

df_patient = pd.read_csv('./data/ptbxl_database.csv')
df_patient = df_patient[df_patient['strat_fold']==10]

patient_filenames = {}
for index, row in df_patient.iterrows():
    patient_id = row['patient_id']
    filename_500 = row['filename_hr']
    filename_500 = filename_500.replace('/', '-')
    if patient_filenames.get(patient_id, None) is None:
        patient_filenames[patient_id] = [filename_500]
    else:
        patient_filenames[patient_id].append(filename_500)

patient_filenames_id = {}

for key in patient_filenames.keys():
    tmpt = patient_filenames[key]
    tmpt_id = []
    for i in range(len(tmpt)):
        tmpt_id.append(filenames_dic[tmpt[i]])
    patient_filenames_id[key] = tmpt_id

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# the data of echonext paper
def bootstrap_score(y_true, y_pred, prc=False, n=1000, base_fn=None):
    if base_fn is None:
        base_fn = sklearn.metrics.average_precision_score if prc else sklearn.metrics.roc_auc_score
    fn = lambda a, b: base_fn(a, b) * 100
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = fn(y_true, y_pred)
    bootstrap_scores = []
    rng = np.random.default_rng(0)
    # replace this
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true))
        # for small number in ptbxl
        while y_true[idx].sum() == 0:
            idx = rng.choice(len(y_true), len(y_true))
        bootstrap_scores.append(fn(y_true[idx], y_pred[idx]))

    lower = np.percentile(bootstrap_scores, 2.5)
    upper = np.percentile(bootstrap_scores, 97.5)
    return score, lower, upper, f'{score:4.1f} ({lower:4.1f}-{upper:4.1f})'

def bootstrap_f1_score(y_true_val, y_pred_proba_val, y_true_test, y_pred_proba_test, n=1000, random_state=0):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true_val, y_pred_proba_val)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    y_pred_binary_test = (y_pred_proba_test >= best_threshold).astype(int)
    score = f1_score(y_true_test, y_pred_binary_test) * 100

    bootstrap_scores = []
    rng = np.random.default_rng(random_state)

    for _ in range(n):
        idx = rng.choice(len(y_true_test), len(y_true_test))

        # for small number in ptbxl
        while y_true_test[idx].sum() == 0:
            idx = rng.choice(len(y_true_test), len(y_true_test))

        y_true_boot = y_true_test[idx]
        y_pred_proba_boot = y_pred_proba_test[idx]
        y_pred_binary_boot = (y_pred_proba_boot >= best_threshold).astype(int)

        if len(np.unique(y_true_boot)) >= 2:
            boot_score = f1_score(y_true_boot, y_pred_binary_boot) * 100
            bootstrap_scores.append(boot_score)
        else:
            bootstrap_scores.append(0.0)

    lower = np.percentile(bootstrap_scores, 2.5)
    upper = np.percentile(bootstrap_scores, 97.5)

    result_str = f'{score:4.1f} ({lower:4.1f}-{upper:4.1f})'

    return score, lower, upper, result_str, best_threshold# the data of echonext paper


def patient_level_id(rng,patient_filenames_id=patient_filenames_id):
    # replace this
    patients = list(patient_filenames_id.keys())
    sampled_patients = rng.choice(
        patients,
        size=len(patients),
        replace=True
    )
    idx = []
    for pid in sampled_patients:
        idx.extend(patient_filenames_id[pid])
    idx = np.array(idx)
    return idx


def bootstrap_score_patient_level(y_true, y_pred, prc=False, n=1000, base_fn=None):
    if base_fn is None:
        base_fn = sklearn.metrics.average_precision_score if prc else sklearn.metrics.roc_auc_score
    fn = lambda a, b: base_fn(a, b) * 100
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = fn(y_true, y_pred)
    bootstrap_scores = []
    rng = np.random.default_rng(0)

    for _ in range(n):
        # for small number in ptbxl
        idx = patient_level_id(rng)
        while y_true[idx].sum() == 0:
            idx = patient_level_id(rng)
        bootstrap_scores.append(fn(y_true[idx], y_pred[idx]))

    lower = np.percentile(bootstrap_scores, 2.5)
    upper = np.percentile(bootstrap_scores, 97.5)
    return score, lower, upper, f'{score:4.1f} ({lower:4.1f}-{upper:4.1f})'

def bootstrap_f1_score_patient_level(y_true_val, y_pred_proba_val, y_true_test, y_pred_proba_test, n=1000, random_state=0):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true_val, y_pred_proba_val)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    y_pred_binary_test = (y_pred_proba_test >= best_threshold).astype(int)
    score = f1_score(y_true_test, y_pred_binary_test) * 100

    bootstrap_scores = []
    rng = np.random.default_rng(random_state)

    for _ in range(n):
        idx = patient_level_id(rng)
        # for small number in ptbxl
        while y_true_test[idx].sum() == 0:
            idx = patient_level_id(rng)

        y_true_boot = y_true_test[idx]
        y_pred_proba_boot = y_pred_proba_test[idx]
        y_pred_binary_boot = (y_pred_proba_boot >= best_threshold).astype(int)

        if len(np.unique(y_true_boot)) >= 2:
            boot_score = f1_score(y_true_boot, y_pred_binary_boot) * 100
            bootstrap_scores.append(boot_score)
        else:
            bootstrap_scores.append(0.0)

    lower = np.percentile(bootstrap_scores, 2.5)
    upper = np.percentile(bootstrap_scores, 97.5)

    result_str = f'{score:4.1f} ({lower:4.1f}-{upper:4.1f})'

    return score, lower, upper, result_str, best_threshold


label_names = {'NORM': 0, 'LVOLT': 1, 'SR': 2, 'SBRAD': 3, 'ABQRS': 4, 'IMI': 5, 'SARRH': 6, 'AFLT': 7, 'AFIB': 8,
               'NDT': 9, 'NST_': 10, 'DIG': 11, 'LVH': 12, 'LPFB': 13, 'LNGQT': 14, 'LAFB': 15, 'RAO/RAE': 16,
               'IRBBB': 17, 'RVH': 18, 'IVCD': 19, 'LMI': 20, 'ASMI': 21, 'AMI': 22, '1AVB': 23, 'ISCAL': 24,
               'STACH': 25, 'ISC_': 26, 'PACE': 27, 'ISCLA': 28, 'SEHYP': 29, 'ISCIL': 30, 'ILMI': 31, 'PVC': 32,
               'CRBBB': 33, 'CLBBB': 34, 'ANEUR': 35, 'ALMI': 36, 'ISCAS': 37, 'TAB_': 38, 'HVOLT': 39, 'LOWT': 40,
               'STD_': 41, 'PAC': 42, 'EL': 43, 'NT_': 44, 'QWAVE': 45, 'INVT': 46, 'LPR': 47, 'VCLVH': 48,
               'LAO/LAE': 49, 'ILBBB': 50, 'ISCIN': 51, 'SVTAC': 52, 'INJAL': 53, 'INJAS': 54, 'IPMI': 55, 'WPW': 56,
               'ISCAN': 57, 'INJLA': 58, 'BIGU': 59, 'TRIGU': 60, 'IPLMI': 61, '3AVB': 62, 'INJIL': 63, '2AVB': 64,
               'PRC(S)': 65, 'PSVT': 66, 'PMI': 67, 'STE_': 68, 'INJIN': 69, 'SVARR': 70}

names_feature_i = {}
for key in label_names.keys():
    names_feature_i[label_names[key]] = key





import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
ptb_res_path = "./saved_models/st_mem_pt"
res_ptb = torch.load(os.path.join(ptb_res_path,'best-MultilabelAUROC.pth'), map_location=torch.device('cpu'))

y_pred_ptb = res_ptb['y_pred_test'].numpy().astype(np.float32)
y_true_ptb = res_ptb['y_true_test'].numpy()
y_pred_val_ptb = np.float32(res_ptb['y_pred_val'].numpy())
y_true_val_ptb = res_ptb['y_true_val'].numpy()


# Bootstrap evaluation
results = []
for feature_i in range(71):

    auc_score_model_ptb, _, _, auc_tmpt_model_ptb = bootstrap_score(y_true_ptb[:,feature_i], y_pred_ptb[:, feature_i], n=1000, prc=False)
    auprc_score_model_ptb, _, _, auprc_tmpt_model_ptb = bootstrap_score(y_true_ptb[:,feature_i], y_pred_ptb[:, feature_i], n=1000, prc=True)
    f1_score_model_ptb, _, _, f1_tmpt_model_ptb, best_threshold_ptb = bootstrap_f1_score(y_true_val_ptb[:,feature_i], y_pred_val_ptb[:,feature_i], y_true_ptb[:,feature_i], y_pred_ptb[:, feature_i], n=1000, random_state=42)

    auc_score_model_ptb_plevel, _, _, auc_tmpt_model_ptb_plevel = bootstrap_score_patient_level(y_true_ptb[:,feature_i], y_pred_ptb[:, feature_i], n=1000, prc=False)
    auprc_score_model_ptb_plevel, _, _, auprc_tmpt_model_ptb_plevel = bootstrap_score_patient_level(y_true_ptb[:,feature_i], y_pred_ptb[:, feature_i], n=1000, prc=True)
    f1_score_model_ptb_plevel, _, _, f1_tmpt_model_ptb_plevel, best_threshold_ptb_plevel = bootstrap_f1_score_patient_level(y_true_val_ptb[:,feature_i], y_pred_val_ptb[:,feature_i], y_true_ptb[:,feature_i], y_pred_ptb[:, feature_i], n=1000, random_state=42)

    if feature_i not in [0,2]:
        auc_score_model, auc_lower_model, auc_upper_model, auc_tmpt_model = bootstrap_score(y_test_j, X_test[:,feature_i], n=1000, prc=False)
        auprc_score_model, auprc_lower_model, auprc_upper_model, auprc_tmpt_model = bootstrap_score(y_test_j, X_test[:,feature_i], n=1000, prc=True)
        f1_score_model, f1_lower_model, f1_upper_model, f1_tmpt_model, best_threshold = bootstrap_f1_score(y_val_j, X_val[:,feature_i], y_test_j, X_test[:,feature_i], n=1000, random_state=42)
        acc_model = accuracy_score(y_true=y_test_j, y_pred=(X_test[:,feature_i] >= best_threshold).astype(int)) * 100

        auc_score_model_val, auc_lower_model_val, auc_upper_model_val, auc_tmpt_model_val = bootstrap_score(y_val_j, X_val[:,feature_i], n=1000, prc=False)
        auprc_score_model_val, auprc_lower_model_val, auprc_upper_model_val, auprc_tmpt_model_val = bootstrap_score(y_val_j, X_val[:,feature_i], n=1000, prc=True)
        f1_score_model_val, f1_lower_model_val, f1_upper_model_val, f1_tmpt_model_val, best_threshold_val = bootstrap_f1_score(y_val_j, X_val[:,feature_i], y_val_j, X_val[:,feature_i], n=1000, random_state=42)
        acc_model_val = accuracy_score(y_true=y_val_j, y_pred=(X_val[:,feature_i] >= best_threshold_val).astype(int)) * 100



    else:
        auc_score_model, auc_lower_model, auc_upper_model, auc_tmpt_model = bootstrap_score(y_test_j, -X_test[:,feature_i], n=1000, prc=False)
        auprc_score_model, auprc_lower_model, auprc_upper_model, auprc_tmpt_model = bootstrap_score(y_test_j, -X_test[:,feature_i], n=1000, prc=True)
        f1_score_model, f1_lower_model, f1_upper_model, f1_tmpt_model, best_threshold = bootstrap_f1_score(y_val_j, -X_val[:,feature_i], y_test_j, -X_test[:,feature_i], n=1000, random_state=42)
        acc_model = accuracy_score(y_true=y_test_j, y_pred=(X_test[:,feature_i] >= best_threshold).astype(int)) * 100
        auc_score_model_val, auc_lower_model_val, auc_upper_model_val, auc_tmpt_model_val = bootstrap_score(y_val_j, -X_val[:,feature_i], n=1000, prc=False)
        auprc_score_model_val, auprc_lower_model_val, auprc_upper_model_val, auprc_tmpt_model_val = bootstrap_score(y_val_j, -X_val[:,feature_i], n=1000, prc=True)
        f1_score_model_val, f1_lower_model_val, f1_upper_model_val, f1_tmpt_model_val, best_threshold_val = bootstrap_f1_score(y_val_j, -X_val[:,feature_i], y_val_j, -X_val[:,feature_i], n=1000, random_state=42)
        acc_model_val = accuracy_score(y_true=y_val_j, y_pred=(X_val[:,feature_i] >= best_threshold).astype(int)) * 100

    print(f"AUROC: {auc_tmpt_model}")
    print(f"AUPRC: {auprc_tmpt_model}")
    print(f"f1   : {f1_tmpt_model}")
    print(f"acc: {acc_model:.1f}%")


    result_row = {
        'feature_i': feature_i,
        'name': names_feature_i[feature_i],
        # 'auroc': float(round(auc_score_model,1)),
        # 'auprc': float(round(auprc_score_model,1)),
        'f1': float(round(f1_score_model,1)),
        # 'f1_val': float(round(f1_score_model_val, 1)),
        'auroc_val': float(round(auc_score_model_val,3)),
        # 'auroc2': float(round(auc_score_model_val, 1)),
        # 'acc_model': round(acc_model, 1),
        'auc_model': auc_tmpt_model,
        'auprc_model': auprc_tmpt_model,
        'f1_model': f1_tmpt_model,
        'best_threshold': best_threshold,
        # 'auc_ptb': auc_score_model_ptb,
        # 'auprc_ptb': auprc_score_model_ptb,
        # 'f1_ptb': f1_score_model_ptb,
        'auc_ptb': auc_tmpt_model_ptb,
        'auprc_ptb': auprc_tmpt_model_ptb,
        'f1_ptb': f1_tmpt_model_ptb,
        'best_threshold_ptb': best_threshold_ptb,
        'auc_ptb_plevel': auc_tmpt_model_ptb_plevel,
        'auprc_ptb_plevel': auprc_tmpt_model_ptb_plevel,
        'f1_ptb_plevel': f1_tmpt_model_ptb_plevel,
        'best_threshold_ptb_plevel': best_threshold_ptb_plevel,
    }
    results.append(result_row)


df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by='auroc_val', ascending=False)
df_sorted.to_csv('./res/results_71preditors_patient_level.csv', index=False)
print(f"\nThe results are saving in results_71preditors.csv， {len(results)}-rows")

