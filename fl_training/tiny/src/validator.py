from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator(object):
    def __init__(self):
        self.metric_score = 0.0

    def validate_loop(self, model, data_loader) -> Dict[str, Any]:
        # Run inference over whole validation set
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True):
                total, correct = 0, 0
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    batch["image"] = batch["image"].to("cuda:0")
                    batch["label"] = batch["label"].to("cuda:0")
                    batch["preds"] = model(batch["image"])
                    _, pred_label = torch.max(batch["preds"].data, 1)
                    total += batch["label"].size(0)
                    correct += (pred_label == batch["label"].data).sum().item()
                    self.metric_score = correct / float(total)

        metrics = {"accuracy": self.metric_score}
        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        return self.validate_loop(model, data_loader)
