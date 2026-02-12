from numpy.random import seed
seed(364)

try:
    from tensorflow import set_random_seed
    set_random_seed(364)
except Exception:
    import tensorflow as tf
    tf.random.set_seed(364)

from parameter_search import parameter_search
from objective_functions import objective_function_stance_branchLSTM_RumEv
from evaluation_functions import evaluation_function_stance_branchLSTM_RumEv

import json
from sklearn.metrics import f1_score, accuracy_score


def label2strA(label):
    if label == 0:
        return "support"
    elif label == 1:
        return "comment"
    elif label == 2:
        return "deny"
    elif label == 3:
        return "query"
    else:
        raise ValueError(f"Unknown stance label: {label}")


def save_stance_json(idsA, predictionsA, out_path="output/answer.json"):
    subtaskaenglish = {id_: label2strA(predictionsA[i]) for i, id_ in enumerate(idsA)}
    answer = {"subtaskaenglish": subtaskaenglish}
    with open(out_path, "w") as f:
        json.dump(answer, f)
    return answer


print("Rumour Stance classification (Task A)")

ntrials = 20
task = "stance"

paramsA, trialsA = parameter_search(ntrials, objective_function_stance_branchLSTM_RumEv, task)

best_trial_id = trialsA.best_trial["tid"]
dev_pred = trialsA.attachments[f"ATTACH::{best_trial_id}::Predictions"]
dev_label = trialsA.attachments[f"ATTACH::{best_trial_id}::Labels"]

print("Dev accuracy:", accuracy_score(dev_label, dev_pred))
print("Dev macro-F1:", f1_score(dev_label, dev_pred, average="macro"))

test_ids, test_pred = evaluation_function_stance_branchLSTM_RumEv(paramsA)

a = save_stance_json(test_ids, test_pred)
print("Saved:", "output/answer.json")
