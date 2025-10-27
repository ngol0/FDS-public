import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import (AlgorithmConstants, AppConstants,
                                             ModelName, ValidateType)

from torch.utils.tensorboard.writer import SummaryWriter
from prettytable import PrettyTable

from utils import extract_weights, load_weights, seed_everything

from ....data_modules import TinyImageNetDataModule
from trainers import get_trainer
from validator import Validator


class TinyImageNetLearner(Learner):
    def __init__(
        self,
        config_train_name: str,
        method,
        seed,
        max_retry=1,
        central=False,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        hyperion=False
    ):
        super().__init__()
        self.config_train_name = config_train_name
        self.method = method
        self.seed = seed
        self.max_retry = max_retry
        self.central = central
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name
        self.hyperion = hyperion

    def initialize(self, parts, fl_ctx):

        seed_everything(self.seed)
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        prefix = Path(self.app_root) if self.app_root else Path.cwd()
        self.writer = SummaryWriter(prefix / "logs", flush_secs=5)
        with open(prefix / "config" / self.config_train_name) as f:
            config_train = json.load(f)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if self.device == "cpu":
            raise ValueError("No GPU detected")
        self.model = eval(
            config_train["model_name"] + f"({self.seed})"
        ).to(self.device)

        if self.hyperion:
            data_dir = os.getenv("HYPERION_TINY_ROOT")
        else:
            data_dir = os.getenv("TINY_ROOT")

        self.dm = TinyImageNetDataModule(
            data_dir=data_dir,
            client_idx=fl_ctx.get_identity_name(),
            central=self.central,
            batch_size=config_train["batch_size"],
            seed=self.seed
        )

        self.local_steps = config_train["local_steps"]
        self.key_metric = "balanced_accuracy"
        self.best_metric = 0.0
        self.best_model_path = "models/best_model.pt"
        self.last_model_path = "models/last.pt"
        self.trainer = (
            get_trainer(self.method, config_train, model=self.model)
            if self.method == "scaffold"
            else get_trainer(self.method, config_train)
        )
        self.validator = Validator()

    def train(self, data, fl_ctx, abort_signal) -> Shareable:

        #  get round information
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        num_rounds = data.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{num_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # SCAFFOLD related
        if self.method == 'scaffold':
            dxo_collection = from_shareable(data)
            if dxo_collection.data_kind != DataKind.COLLECTION:
                self.log_error(
                    fl_ctx,
                    f"SCAFFOLD learner expected shareable to contain a collection of two DXOs "
                    f"but got data kind {dxo_collection.data_kind}.",
                )
                return make_reply(ReturnCode.ERROR)
            dxo_global_weights = dxo_collection.data.get(AppConstants.MODEL_WEIGHTS)
            dxo_global_ctrl_weights = dxo_collection.data.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
            if dxo_global_ctrl_weights is None:
                self.log_error(
                    fl_ctx, "DXO collection doesn't contain the SCAFFOLD controls!"
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            # convert to tensor and load into c_global model
            global_ctrl_weights = dxo_global_ctrl_weights.data
            for k in global_ctrl_weights.keys():
                global_ctrl_weights[k] = torch.as_tensor(global_ctrl_weights[k])
            self.trainer.scaffold_helper.load_global_controls(weights=global_ctrl_weights)

            # modify shareable to only contain global weights so normal training can proceed
            data = dxo_global_weights.update_shareable(data)

        # update local model weights with received weights
        dxo = from_shareable(data)
        global_weights = dxo.data

        # NB load_weights(self.model, dxo.data) loads the weights into the model during validation
        # as the sequence is validate, train, validate, train, ...

        # Run training
        for i in range(self.max_retry + 1):
            try:
                self.trainer.run(
                    self.model,
                    self.dm.train_dataloader,
                    num_steps=self.local_steps,
                    logger=self.writer,
                    device=self.device,
                    fl_ctx=fl_ctx,
                )
                break
            except Exception as e:
                if i < self.max_retry:
                    self.log_warning(fl_ctx, "Something wrong in training, retrying.")
                    # Restore trainer states to the beginning of the round
                    if os.path.exists(self.last_model_path):
                        self.log_warning(fl_ctx, f"Loading last model from {self.last_model_path}")
                        self.trainer.load_checkpoint(self.last_model_path, self.model, fl_ctx=fl_ctx)
                        load_weights(self.model, global_weights)
                        self.model = self.model.to(self.device)
                    # Reset dataset & dataloader
                    self.dm.teardown()
                    self.dm.setup()
                else:
                    raise RuntimeError(traceback.format_exc())

        # Run post-training validation
        for i in range(self.max_retry + 1):
            try:
                metrics = self.validator.run(self.model, self.dm.val_dataloader)
                break
            except Exception as e:
                if i < self.max_retry:
                    self.log_warning(fl_ctx, f"Something wrong in val, retrying ({i+1}/{self.max_retry}).")
                    # Reset dataset & dataloader
                    self.dm.teardown()
                    self.dm.setup()
                else:
                    raise RuntimeError(traceback.format_exc())

        # Log validation results
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for m, v in metrics.items():
            table.add_row([m, v])
            self.writer.add_scalar(f"Accuracy/Post-Train Val Accuracy", v, self.trainer.current_step)
        self.log_info(fl_ctx, str(table))

        # Save checkpoint if necessary
        if self.best_metric < metrics[self.key_metric]:
            self.best_metric = metrics[self.key_metric]
            self.trainer.save_checkpoint(self.best_model_path, self.model)
        self.trainer.save_checkpoint(self.last_model_path, self.model)

        # Calculate weight diff
        local_weights = extract_weights(self.model)
        weight_diff = {}
        for var_name in local_weights:
            weight_diff[var_name] = local_weights[var_name] - global_weights[var_name]
            if np.any(np.isnan(weight_diff[var_name])):
                self.system_panic(f"{var_name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Create DXO and return
        dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=weight_diff,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.local_steps},
        )

        # SCAFFOLD related
        if self.method == 'scaffold':
            # Create a DXO collection with weights and scaffold controls
            dxo_weights_diff_ctrl = DXO(
                data_kind=DataKind.WEIGHT_DIFF,
                data=self.trainer.scaffold_helper.get_delta_controls(),
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.local_steps},
            )
            collection_data = {
                    AppConstants.MODEL_WEIGHTS: dxo,
                    AlgorithmConstants.SCAFFOLD_CTRL_DIFF: dxo_weights_diff_ctrl,
                }

            dxo = DXO(data_kind=DataKind.COLLECTION, data=collection_data)

        return dxo.to_shareable()

    def validate(self, data, fl_ctx, abort_signal):

        model_owner = data.get_header(AppConstants.MODEL_OWNER, "global_model")
        validate_type = data.get_header(AppConstants.VALIDATE_TYPE)
        # 2. Prepare dataset
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            data_loader = self.dm.val_dataloader
        elif validate_type == ValidateType.MODEL_VALIDATE:
            # test set here if used
            data_loader = self.dm.val_dataloader
        # update local model weights with received weights
        try:
            dxo = from_shareable(data)
        except Exception as e:
            self.log_error(fl_ctx, "Error when extracting DXO from shareable")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # SCAFFOLD learner expects shareable to contain a collection of two DXOs
        if dxo.data_kind == DataKind.COLLECTION:
            assert self.method == "scaffold"
            # strip out the scaffold controls
            dxo = dxo.data.get(AppConstants.MODEL_WEIGHTS)

        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # load global weights to model
        load_weights(self.model, dxo.data)

        self.model = self.model.to(self.device)
        for i in range(self.max_retry + 1):
            try:
                raw_metrics = self.validator.run(self.model, data_loader)
                break
            except Exception as e:
                if i < self.max_retry:
                    self.log_warning(
                        fl_ctx, "Error encountered in validation, retrying."
                    )
                    # Cleanup previous dataset & dataloader
                    self.dm.teardown()
                    self.dm.setup()
                else:
                    raise RuntimeError(traceback.format_exc())

        self.log_info(
            fl_ctx,
            f"Validation metrics of {model_owner}'s model on" f" {fl_ctx.get_identity_name()}'s data: {raw_metrics}",
        )

        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            metrics = {MetaKey.INITIAL_METRICS: raw_metrics[self.key_metric]}
            # Save as best model
            if self.best_metric < raw_metrics[self.key_metric]:
                self.best_metric = raw_metrics[self.key_metric]
                self.trainer.save_checkpoint(self.best_model_path, self.model)
        else:
            metrics = raw_metrics

        # 5. Return results
        dxo = DXO(data_kind=DataKind.METRICS, data=metrics)
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        if model_name == ModelName.BEST_MODEL or model_name == ModelName.FINAL_MODEL:
            model_data = None
            try:
                if model_name == ModelName.FINAL_MODEL:
                    model_data = torch.load(self.last_model_path, map_location="cpu")
                    self.log_info(fl_ctx, f"Load final model from {self.last_model_path}")
                else:
                    model_data = torch.load(self.best_model_path, map_location="cpu")
                    self.log_info(fl_ctx, f"Load best model from {self.best_model_path}")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load model: {e}")

            if model_data:
                data = {}
                for var_name in model_data["model"]:
                    data[var_name] = model_data["model"][var_name].numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
                return dxo.to_shareable()
            else:
                self.log_error(fl_ctx, "local model not available")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            self.log_error(fl_ctx, f"Unknown model_type {model_name}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

    def finalize(self, fl_ctx):
        self.dm.teardown()
        self.writer.close()
