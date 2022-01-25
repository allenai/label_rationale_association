"""
Trainer class used by training scripts.
Heavily based on https://github.com/huggingface/transformers/tree/7cb203fae4e7964e9e99400b375d660ebce765ee/src/transformers/trainer.py (Huggingface Transformers v2.9.1)
See Huggingface repository for licensing agreement.

Code formatted using https://github.com/psf/black
"""

import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
)
from transformers.training_args import TrainingArguments
from transformers import T5ForConditionalGeneration
from custom_args import DataTrainingArguments

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_args: DataTrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_args: DataTrainingArguments,
        data_collator: DataCollator,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        set_seed(self.args.seed)
        # Create output directory if needed
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.early_stopping_threshold = data_args.early_stopping_threshold
        self.log = os.path.join(self.args.output_dir, "training_log.txt")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = RandomSampler(self.train_dataset)

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self):
        """
        Main training entry point.
        """
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = int(
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)
        model = self.model

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size * self.args.gradient_accumulation_steps
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_gpu_train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. accumulation) = %d",
            total_train_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0

        dev_losses = []
        best_epoch = num_train_epochs
        model.zero_grad()
        train_iterator = trange(
            epochs_trained,
            int(num_train_epochs),
            desc="Epoch",
        )
        for epoch in train_iterator:

            # reset training loss per-epoch
            tr_loss = []

            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
            )

            for step, inputs in enumerate(epoch_iterator):

                tr_loss.append(self._training_step(model, inputs, optimizer))

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = round(epoch + (step + 1) / len(epoch_iterator))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            # do saving & logging once epoch is complete
            if (
                self.args.logging_steps > 0
                and self.epoch % self.args.logging_steps == 0
            ) or (self.epoch == 1 and self.args.logging_first_step):
                logs: Dict[str, float] = {}
                # compute avg. epoch loss
                logs["loss"] = np.mean(tr_loss)
                logger.info("Training loss: %0.4f" % logs["loss"])
                # backward compatibility for pytorch schedulers
                logs["learning_rate"] = (
                    scheduler.get_last_lr()[0]
                    if version.parse(torch.__version__) >= version.parse("1.4")
                    else scheduler.get_lr()[0]
                )

                self._log(logs)

                # evaluate dev_loss on the dev set
                self.evaluate()
            else:
                logger.info("not logging training loss")

            if self.args.save_steps > 0 and self.epoch % self.args.save_steps == 0:
                # self.model is always a reference to the model we want to save.
                if hasattr(model, "module"):
                    assert model.module is self.model
                else:
                    assert model is self.model
                # Save model checkpoint
                output_dir = os.path.join(
                    self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(self.epoch)}"
                )

                self.save_model(output_dir)
                self._rotate_checkpoints()

                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            else:
                logger.info("not saving model")

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

            # check if early stopping satisfied
            curr_dev_loss = self.evaluate(self.eval_dataset)["eval_loss"]
            dev_losses.append(curr_dev_loss)
            if dev_losses.index(min(dev_losses)) < (
                len(dev_losses) - self.early_stopping_threshold
            ):
                logger.info(
                    "Early stopping at epoch %d and saving model...dev loss has not decreased for %d epochs"
                    % (epoch + 1, self.early_stopping_threshold)
                )
                logger.info(
                    "Best epoch = %d" % (epoch + 1 - self.early_stopping_threshold)
                )
                logger.info("Dev loss at best epoch: %0.4f" % min(dev_losses))

                # set self.model to the best model so that subsequent evaluation is from this
                best_epoch = epoch + 1 - self.early_stopping_threshold
                directory = os.path.join(
                    self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{int(best_epoch)}"
                )
                logger.info("Reloading best-epoch model...")
                self.model = T5ForConditionalGeneration.from_pretrained(directory).to(
                    self.args.device
                )

                # stop training
                break

        logger.info("\n\nTraining completed\n\n")

        # delete extra checkpoints
        ckpts = [
            name
            for name in os.listdir(self.args.output_dir)
            if PREFIX_CHECKPOINT_DIR in name
        ]
        keep_model = f"{PREFIX_CHECKPOINT_DIR}-{int(best_epoch)}"
        ckpts.remove(keep_model)
        for el in ckpts:
            shutil.rmtree(os.path.join(self.args.output_dir, el))

        return TrainOutput(self.epoch, np.mean(tr_loss))

    def _log(self, logs: Dict[str, float]) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        output = json.dumps({**logs, **{"step": self.epoch}})
        with open(self.log, "a") as f:
            f.write(output)
            f.write("\n")

    def _training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float:

        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        # model outputs are a tuple
        loss = outputs[0]

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.item()

    def save_model(self, output_dir: Optional[str] = None):

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(
        self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [
            str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")
        ]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path)
                    )

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - self.args.save_total_limit
        )
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            shutil.rmtree(checkpoint)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.prediction_loss_only
        )

        model = self.model
        batch_size = dataloader.batch_size
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = inputs.get("lm_labels") is not None

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("lm_labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["lm_labels"].detach()
                    else:
                        label_ids = torch.cat(
                            (label_ids, inputs["lm_labels"].detach()), dim=0
                        )

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids)
            )
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        logger.info("%s loss: %0.4f" % (description, metrics["eval_loss"]))

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
