import json
from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

# The three class-weighting variants for nn.CrossEntropyLoss, directly
# analogous to the LDA prior_mode sweep ("empirical" / "uniform" / "balanced").
WEIGHT_MODES = ["empirical", "uniform", "balanced"]

def _weight_modes_for(args) -> list:
    """
    Which weight_modes to sweep in train() for this run. Defaults to the
    full WEIGHT_MODES, with two ways to narrow it:
      1. An explicit args.weight_modes (list of strings), if the CLI/run.py
         ever exposes one -- takes priority.
      2. AMP-SCZ specifically: restricted to ["uniform", "balanced"] --
         "empirical" (no reweighting) is skipped for this dataset per
         request, without needing any run.py/argparse changes. Detected
         from args.data and args.model_id (whichever is set) containing
         both "AMP" and "SCZ", case-insensitively.
    """
    override = getattr(args, "weight_modes", None)
    if override:
        return list(override)

    name = (str(getattr(args, "data", "")) + str(getattr(args, "model_id", ""))).upper()
    if ("AMP" in name and "SCZ" in name) or ("TDBRAIN" in name) or ("APAVA" in name):
        return ["uniform", "balanced"]
    elif ("ADFTD" in name):
        return ["empirical", "balanced"]

    return WEIGHT_MODES


def compute_class_weights(ytr, mode, n_classes):
    """
    Mirrors the LDA prior_mode formulas:
      - "empirical": no weighting at all (None) -- the natural training
        distribution is what CrossEntropyLoss sees by default.
      - "uniform":   equal weight per class, regardless of frequency.
      - "balanced":  inverse-frequency weights, normalized to sum to 1
        (same formula as `priors = (1.0/counts) / (1.0/counts).sum()`).
    """
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0  # avoid divide-by-zero for a class absent in this split

    if mode == "empirical":
        return None
    elif mode == "uniform":
        return np.ones(n_classes) / n_classes
    elif mode == "balanced":
        inv = 1.0 / counts
        return inv / inv.sum()
    else:
        raise ValueError(f"Unknown weight mode: {mode}")


def unpack_batch(batch):
    """
    Supports batches shaped like:
      (x, y)
      (x, y, padding_mask)
      (x, y, padding_mask, subj_id)

    Returns:
      x, y, padding_mask_or_None, subj_id_or_None
    """

    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    if len(batch) == 4:
        batch_x, label, padding_mask, subj_id = batch
        return batch_x, label, padding_mask, subj_id

    elif len(batch) == 3:
        batch_x, label, padding_mask = batch
        return batch_x, label, padding_mask, None

    elif len(batch) == 2:
        batch_x, label = batch
        return batch_x, label, None, None

    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        # Set once train() (or a standalone test(..., test=1)) resolves which
        # weight_mode won on validation F1 for this setting.
        self.selected_weight_mode = None
        self.selected_class_weight = None

    def _build_model(self):
        # model input depends on data
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, class_weight=None):
        if class_weight is not None:
            weight_t = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
            return nn.CrossEntropyLoss(weight=weight_t)
        return nn.CrossEntropyLoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        subj_ids = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, label, padding_mask, subj_id = unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                if padding_mask is not None:
                    padding_mask = padding_mask.float().to(self.device)

                outputs = self.model(batch_x)

                # Compute loss on-device (matches train()'s device usage).
                # Moving pred/label to CPU here while `criterion`'s weight
                # tensor lives on self.device (cuda) is what caused the
                # "Expected all tensors to be on the same device" error —
                # it only surfaced once a weighted mode's criterion carried
                # a real CUDA weight tensor to mismatch against.
                pred = outputs.detach()
                loss = criterion(pred, label.long())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

                if subj_id is None:
                    raise RuntimeError(
                        "Subject-level eval requested but dataset did not return subj_id. "
                        "Make sure __getitem__ returns (x, y, padding_mask, subj_id)."
                    )
                if torch.is_tensor(subj_id):
                    subj_ids.append(subj_id.detach().cpu())
                else:
                    # list/tuple of ids (e.g., strings) from default collate
                    subj_ids.extend(list(subj_id))

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        trues_np = trues.flatten().cpu().numpy()

        if torch.is_tensor(subj_ids[0]) if len(subj_ids) > 0 else False:
            subj_ids = torch.cat(subj_ids, 0).numpy()
        else:
            subj_ids = np.array(subj_ids)

        # --- subject-level aggregation (soft voting) ---
        unique_subj = np.unique(subj_ids)
        subj_probs = np.zeros((len(unique_subj), self.args.num_class), dtype=np.float32)
        subj_trues = np.zeros(len(unique_subj), dtype=np.int64)
        for k, s in enumerate(unique_subj):
            mask = subj_ids == s
            subj_probs[k] = probs[mask].mean(axis=0)
            # all windows of a subject must share the same label
            labels_for_subj = trues_np[mask]
            assert (labels_for_subj == labels_for_subj[0]).all(), \
                f"Inconsistent labels within subject {s}"
            subj_trues[k] = labels_for_subj[0]

        subj_preds = np.argmax(subj_probs, axis=1)
        subj_trues_onehot = np.eye(self.args.num_class)[subj_trues]

        metrics_dict = {
            "Accuracy":  accuracy_score(subj_trues, subj_preds),
            "Precision": precision_score(subj_trues, subj_preds, average="macro", zero_division=0),
            "Recall":    recall_score(subj_trues, subj_preds, average="macro", zero_division=0),
            "F1":        f1_score(subj_trues, subj_preds, average="macro", zero_division=0),
            "AUROC":     roc_auc_score(subj_trues_onehot, subj_probs, multi_class="ovr"),
            "AUPRC":     average_precision_score(subj_trues_onehot, subj_probs, average="macro"),
        }

        self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        setting_root = "./checkpoints/" + self.args.model + "/" + setting + "/"
        os.makedirs(setting_root, exist_ok=True)

        ytr       = np.asarray(train_data.y).astype(int).reshape(-1)
        n_classes = self.args.num_class

        # Snapshot the model's initial (random) weights once, so every
        # weight_mode below trains from the same starting point rather than
        # continuing from wherever the previous mode's training left off.
        initial_state = deepcopy(self.model.state_dict())

        time_now    = time.time()
        train_steps = len(train_loader)

        # mode -> {"best_val_f1": float, "checkpoint_path": str, "class_weight": list|None}
        mode_results = {}
        
        weight_modes = _weight_modes_for(self.args)
        print(f"[weight_modes] sweeping: {weight_modes}")

        for mode in weight_modes:
            print(f"\n{'=' * 20} Training with weight_mode={mode} {'=' * 20}")
            class_weight = compute_class_weights(ytr, mode, n_classes)
            criterion    = self._select_criterion(class_weight)

            self.model.load_state_dict(deepcopy(initial_state))
            model_optim = self._select_optimizer()

            mode_path = setting_root + mode + "/"
            os.makedirs(mode_path, exist_ok=True)

            early_stopping        = EarlyStopping(patience=self.args.patience, verbose=True, delta=1e-5)
            best_val_f1_this_mode = -np.inf

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()

                for i, batch in enumerate(train_loader):
                    batch_x, label, padding_mask, subj_id = unpack_batch(batch)
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    label = label.to(self.device)
                    if padding_mask is not None:
                        padding_mask = padding_mask.float().to(self.device)

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, label.long())
                    train_loss.append(loss.item())

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
                test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

                print(
                    f"[{mode}] Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                    f"Validation results --- Loss: {vali_loss:.5f}, "
                    f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                    f"Precision: {val_metrics_dict['Precision']:.5f}, "
                    f"Recall: {val_metrics_dict['Recall']:.5f}, "
                    f"F1: {val_metrics_dict['F1']:.5f}, "
                    f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                    f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                    f"Test results --- Loss: {test_loss:.5f}, "
                    f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                    f"Precision: {test_metrics_dict['Precision']:.5f}, "
                    f"Recall: {test_metrics_dict['Recall']:.5f} "
                    f"F1: {test_metrics_dict['F1']:.5f}, "
                    f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                    f"AUPRC: {test_metrics_dict['AUPRC']:.5f}"
                )

                if val_metrics_dict["F1"] > best_val_f1_this_mode:
                    best_val_f1_this_mode = val_metrics_dict["F1"]

                early_stopping(-val_metrics_dict["F1"], self.model, mode_path)
                if early_stopping.early_stop:
                    print(f"[{mode}] Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            mode_results[mode] = {
                "best_val_f1":     best_val_f1_this_mode,
                "checkpoint_path": mode_path + "checkpoint.pth",
                "class_weight":    None if class_weight is None else class_weight.tolist(),
            }
            print(f"[{mode}] best validation F1 this mode: {best_val_f1_this_mode:.5f}")

        # --- model selection: whichever weight_mode had the best validation F1 ---
        best_mode = max(mode_results, key=lambda m: mode_results[m]["best_val_f1"])
        print(f"\n[SELECTED] weight_mode={best_mode}  "
              f"(val F1={mode_results[best_mode]['best_val_f1']:.5f})")

        self.model.load_state_dict(torch.load(mode_results[best_mode]["checkpoint_path"]))
        self.selected_weight_mode  = best_mode
        self.selected_class_weight = mode_results[best_mode]["class_weight"]

        # Persist the selection so a standalone test(setting, test=1) call in
        # a fresh process (no train() beforehand) can reproduce exactly this
        # mode+checkpoint rather than defaulting to the wrong one.
        with open(setting_root + "selected_weight_mode.json", "w") as f:
            json.dump({
                "mode":            self.selected_weight_mode,
                "class_weight":    self.selected_class_weight,
                "checkpoint_path": mode_results[best_mode]["checkpoint_path"],
                "all_modes":       mode_results,  # kept for reference/debugging
            }, f, indent=2)

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        setting_root   = "./checkpoints/" + self.args.model + "/" + setting + "/"
        selection_file = setting_root + "selected_weight_mode.json"

        if test or self.selected_weight_mode is None:
            print("loading model")
            if not os.path.exists(selection_file):
                raise Exception(
                    f"No weight-mode selection found at {selection_file}. "
                    f"Run train(setting) first so a mode can be selected."
                )
            with open(selection_file) as f:
                selection = json.load(f)

            model_path = selection["checkpoint_path"]
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            self.model.load_state_dict(torch.load(model_path))

            self.selected_weight_mode  = selection["mode"]
            self.selected_class_weight = selection["class_weight"]

        # Uncomment below code for save space on device
        self.del_weight(setting_root)

        criterion = self._select_criterion(self.selected_class_weight)
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        print(
            f"[weight_mode={self.selected_weight_mode}]\n"
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        return test_metrics_dict

    def del_weight(self, setting_root):
        for mode in WEIGHT_MODES:
            ckpt = os.path.join(setting_root, mode, "checkpoint.pth")
            if os.path.exists(ckpt):
                os.remove(ckpt)
                print(f"Model weights deleted for mode={mode}....")
