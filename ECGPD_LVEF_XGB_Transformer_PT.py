import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
import sklearn.metrics
import random
# load data
from ECG_preprocess_LVEF import (y_train, y_val, y_test, X_train, X_val, X_test,
                                 label_names as shd)


use_index = 0 # corresponds to low LVEF

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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


print("\nXGBoost model training...")

max_depth_range = [3,5,7]
lr_range = [0.05, 0.1, 0.2]


val_auc_best = 0
max_depth_best = 0
lr_best = 0
xgb_model_best = None

# hyper-parameter tuning
for max_depth in max_depth_range:
    for lr in lr_range:
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_estimators=1000,
            max_depth=max_depth,
            learning_rate=lr,
            scale_pos_weight=1.0,
            eval_metric='auc',
            early_stopping_rounds=30
        )

        # model training
        xgb_model.fit(
            X_train,
            y_train_j,
            eval_set=[(X_val, y_val_j)],
            verbose=False
        )

        # validation results
        y_pred_proba_val = xgb_model.predict_proba(X_val)
        auc_val = roc_auc_score(y_val_j, y_pred_proba_val[:, 1])
        auprc_val = average_precision_score(y_val_j, y_pred_proba_val[:, 1])

        if auc_val > val_auc_best:
            val_auc_best = auc_val
            max_depth_best = max_depth
            lr_best = lr
            xgb_model_best = xgb_model



y_pred_proba_val = xgb_model_best.predict_proba(X_val)
auc_val = roc_auc_score(y_val_j, y_pred_proba_val[:, 1])
auprc_val = average_precision_score(y_val_j, y_pred_proba_val[:, 1])

y_pred_proba = xgb_model_best.predict_proba(X_test)
auc = roc_auc_score(y_test_j, y_pred_proba[:, 1])
auprc = average_precision_score(y_test_j, y_pred_proba[:, 1])


print(f"\n=== XGBoost Model evaluation ({shd[use_index]}) ===")
print(f"AUROC: test={auc:.4f} val={auc_val:.4f}")
print(f"AUPRC: test={auprc:.4f} val={auprc_val:.4f}")

################
# evaluation
################
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
    for _ in range(n):
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

# Bootstrap evaluation
auc_score_model, auc_lower_model, auc_upper_model, auc_tmpt_model = bootstrap_score(y_test_j, y_pred_proba[:, 1], n=1000, prc=False)
auprc_score_model, auprc_lower_model, auprc_upper_model, auprc_tmpt_model = bootstrap_score(y_test_j, y_pred_proba[:, 1], n=1000, prc=True)
f1_score_model, f1_lower_model, f1_upper_model, f1_tmpt_model, best_threshold = bootstrap_f1_score(y_val_j, y_pred_proba_val[:, 1], y_test_j, y_pred_proba[:, 1], n=1000, random_state=42)

print(f"AUROC: {auc_tmpt_model}")
print(f"AUPRC: {auprc_tmpt_model}")
print(f"f1   : {f1_tmpt_model}")
print(f"acc: {accuracy_score(y_true=y_test_j, y_pred=(y_pred_proba[:, 1] >= best_threshold).astype(int)) * 100:.1f}%")
print(f"depth: {max_depth_best}, lr: {lr_best}")

# xgb_model_best.save_model('./saved_models/xgboost_transformer_pt.json')
