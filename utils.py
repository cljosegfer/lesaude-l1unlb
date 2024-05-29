
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict

SIGNAL_CROP_LEN = 2560
SIGNAL_NON_ZERO_START = 571

def get_tokens(text, tokenizer, device = 'cuda'):
    encoded = tokenizer.batch_encode_plus(text, return_tensors = "pt", max_length = 256, padding = 'max_length', truncation = True)
    tokens, _, attention = encoded.values()
    return tokens.to(device), attention.to(device)

def plot_epoch(model_label, log_trn, log_val = None, epoch = None):
    plt.figure();
    plt.plot(log_trn);
    if log_val != None:
        plt.axhline(y = log_val, color = 'tab:orange');
        plt.title('trn: {}, val: {}'.format(np.mean(log_trn), log_val));
    plt.title('trn: {}, val: {}'.format(np.mean(log_trn), np.mean(log_val)));
    if epoch != None:
        plt.savefig('output/{}/fig/loss_{}.png'.format(model_label, epoch));
        plt.close();

def plot_log(model_label, log):
    log = np.array(log)
    log_trn = log[:, 0]
    log_val = log[:, 1]

    plt.figure();
    plt.plot(log_trn, color = 'tab:blue');
    plt.plot(log_val, color = 'tab:orange');

    minimo = np.min(log_val)
    plt.axhline(y = minimo, color = 'tab:red');
    plt.title('min val: {}'.format(minimo));

    plt.savefig('output/{}/loss.png'.format(model_label));
    plt.close();

def export(model, model_label, epoch):
    if epoch != None:
        print('exporting partial model at epoch {}'.format(epoch))
        torch.save(model, 'output/{}/partial.pt'.format(model_label))
    else:
        print('exporting last model')
        torch.save(model, 'output/{}/{}.pt'.format(model_label, model_label))

# def get_inputs(batch, apply = "non_zero", device = "cuda"):
def get_inputs(batch, apply = "nothing", device = "cuda"):
    # (B, C, L)
    if batch.shape[1] > batch.shape[2]:
        batch = batch.permute(0, 2, 1)

    B, n_leads, signal_len = batch.shape

    if apply == "non_zero":
        transformed_data = torch.zeros(B, n_leads, SIGNAL_CROP_LEN)
        for b in range(B):
            start = SIGNAL_NON_ZERO_START
            diff = signal_len - start
            if start > diff:
                correction = start - diff
                start -= correction
            end = start + SIGNAL_CROP_LEN
            for l in range(n_leads):
                transformed_data[b, l, :] = batch[b, l, start:end]

    else:
        transformed_data = batch.float()

    return transformed_data.to(device)

def find_best_thresholds(predictions, true_labels_dict, thresholds):
    num_classes = len(predictions[0])
    best_thresholds = [0.5] * num_classes
    best_f1s = [0.0] * num_classes

    for class_idx in (range(num_classes)):
        for thresh in thresholds:
            f1 = f1_score(
                true_labels_dict[class_idx],
                predictions[thresh][class_idx],
                zero_division=0,
            )

            if f1 > best_f1s[class_idx]:
                best_f1s[class_idx] = f1
                best_thresholds[class_idx] = thresh
    
    return best_f1s, best_thresholds

def metrics_table(all_binary_results, all_true_labels):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    num_classes = all_binary_results.shape[-1]
    for class_idx in range(num_classes):
        class_binary_results = all_binary_results[:, class_idx].cpu().numpy()
        class_true_labels = all_true_labels[:, class_idx].cpu().numpy()

        accuracy = accuracy_score(class_true_labels, class_binary_results)
        precision = precision_score(
            class_true_labels, class_binary_results, zero_division=0
        )
        recall = recall_score(
            class_true_labels, class_binary_results, zero_division=0
        )
        f1 = f1_score(class_true_labels, class_binary_results, zero_division=0)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    metrics_dict = {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1 Score": f1_scores,
    }

    return metrics_dict

def json_dump(metrics_dict, model_label):
    with open('output/{}/metrics.json'.format(model_label), 'w') as f:
        json.dump(metrics_dict, f)

def load_backbone(model, backbone_path):
    backbone = torch.load(backbone_path)
    backbone = torch.nn.ModuleList(backbone.children())[:-1]

    key_transformation = []
    for key in model.state_dict().keys():
        key_transformation.append(key)

    state_dict = backbone.state_dict()
    new_state_dict = OrderedDict()
    for i, (key, value) in enumerate(state_dict.items()):
        new_key = key_transformation[i]
        new_state_dict[new_key] = value

    log = model.load_state_dict(new_state_dict, strict = False)
    assert log.missing_keys == ['linear.weight', 'linear.bias']
    
    return {'model': model, 'log': log}