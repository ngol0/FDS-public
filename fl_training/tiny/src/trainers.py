from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import LogMessageTag
from nvflare.apis.utils.fl_context_utils import generate_log_message
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import os
from timm.utils.agc import adaptive_clip_grad

from validator import Validator


class BaseTrainer(FLComponent):

    def __init__(self, config, model=None):
        super().__init__()

        # rename FL logger to avoid conflict with tensorboard logger
        self.fl_logger = self.logger
        self.logger = None

        # Reset/Init step/round and opt/sched attributes
        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None

        self.max_steps = config["max_steps"]
        self.steps_or_epochs = config["steps_or_epochs"]

        self.optimizer_dict = config["optimizer_args"]
        self.optim_resume = config["optim_resume"]
        self.lr = config["learning_rate"]
        self.scheduler_type = config["scheduler"]

        self.grad_clipping_type = config["grad_clipping_type"]
        self.grad_clipping_value = config["grad_clipping_value"]
        self.gradient_accumulation_steps = 1
        self.use_half_precision = config["use_half_precision"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)

        self.criterion = nn.CrossEntropyLoss()

        self.fedprox_mu = config["fedprox_mu"]
        self.criterion_prox = None
        if self.fedprox_mu > 0:
            self.criterion_prox = PTFedProxLoss(mu=self.fedprox_mu)

        self.validator = Validator()

    def log_info(self, fl_ctx, msg, fire_event=True):
        # overload FLComponent log_info method to change logger name
        log_msg = generate_log_message(fl_ctx, msg)
        self.fl_logger.info(log_msg)
        if fire_event:
            self._fire_log_event(
                event_type=EventType.INFO_LOG_AVAILABLE,
                log_tag=LogMessageTag.INFO,
                log_msg=log_msg,
                fl_ctx=fl_ctx,
            )

    def configure_optimizer(self):

        optimizer_class = self.optimizer_dict.pop("class", None)
        self.opt = eval(f"torch.optim.{optimizer_class}")(self.model.parameters(), lr=self.lr, **self.optimizer_dict)
        self.optimizer_dict["class"] = optimizer_class

        # Load optimizer state
        if (self.opt_state is not None) and (self.optim_resume == "resume"):
            self.opt.load_state_dict(self.opt_state)

        # Configure scheduler
        if self.scheduler_type == "onecycle":
            self.sch = torch.optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=self.lr,
                total_steps=self.max_steps,
                anneal_strategy="cos",
                final_div_factor=1e4,
            )
        elif self.scheduler_type == "cosine":
            self.sch = CosineAnnealingLR(self.opt, T_max=self.max_steps, eta_min=0.001)

        elif self.scheduler_type == "multistep":
            self.sch = MultiStepLR(self.opt, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5)

        elif self.scheduler_type == "none":
            self.sch = torch.optim.lr_scheduler.ConstantLR(self.opt, factor=1, total_iters=self.max_steps)
        else:
            raise ValueError(f"Invalid scheduler type {self.scheduler_type}")

        # Load scheduler state
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def get_batch(self, data_loader: DataLoader, num_steps: int):

        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):

        # if steps_or_epochs is epochs, num_steps given to training_loop
        # is the number of epochs, otherwise it is the number of steps
        # the rest of training_loop treats num_steps as actual number of steps
        # so convert num_steps to the number of steps if necessary
        if self.steps_or_epochs == "epochs":
            num_steps = len(data_loader) * num_steps
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.autocast(device_type="cuda", enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps
                    if self.criterion_prox is not None:
                        loss = loss + self.criterion_prox(self.model, self.global_model) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.grad_clipping_type == "norm":
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clipping_value,
                            norm_type=2.0,
                        )
                    elif self.grad_clipping_type == "agc":
                        self.scaler.unscale_(self.opt)
                        adaptive_clip_grad(
                            self.model.parameters(),
                            self.grad_clipping_value,
                        )

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)
                # Write average loss to tensorboard
                avg_loss += loss.item()

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps,
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps,
                )

    def setup(self, model, logger, data_loader, num_steps, abort_signal):

        self.model = model

        # global model setup for aux losses
        if self.fedprox_mu > 0:
            self.global_model = deepcopy(model).eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        if self.steps_or_epochs == "epochs":
            self.max_steps = len(data_loader) * num_steps

        self.configure_optimizer()

    def cleanup(self):

        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict()) if self.optim_resume == "resume" else None
        self.sch_state = deepcopy(self.sch.state_dict())

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:

        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module, fl_ctx=None):

        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
        fl_ctx=None,
    ):

        self.setup(model, logger, data_loader, num_steps, abort_signal)
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1
        self.cleanup()


def get_trainer(method, config, class_weights, **kwargs):
    if method in ["fedavg", "fedprox", "fedopt"]:
        return BaseTrainer(config, class_weights)
    else:
        raise ValueError(f"Non-implemented method {method}")
