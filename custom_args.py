"""
Custom arguments and evaluation helper functions.
Partially based on https://github.com/huggingface/transformers/tree/7cb203fae4e7964e9e99400b375d660ebce765ee/examples/language-modeling/run_language_modeling.py (Huggingface Transformers v2.9.1)
See Huggingface repository for licensing agreement.

Code formatted using https://github.com/psf/black
"""

from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch
import os


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    rationale_only: bool = field(
        default=False,
        metadata={
            "help": "Only produce rationales and not labels (first model in pipeline)"
        },
    )
    label_only: bool = field(
        default=False,
        metadata={"help": "Only produce labels and not rationales (I-->O baseline)"},
    )
    include_input: bool = field(
        default=False,
        metadata={"help": "Append input to second model in pipeline"},
    )
    use_dev_real_expls: bool = field(
        default=False,
        metadata={
            "help": "Use this flag for test case where we want to test on gold-label predictions rather than generations"
        },
    )
    pretrained_model_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pass a pretrained model save_path to re-load for evaluation"
        },
    )
    predictions_model_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pass a file where can find predictions from generation model for the dev set (first model in pipeline)"
        },
    )
    dropout_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": "Specify a dropout rate, if don't want to use default in transformers/configuration_t5.py"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on."})
    early_stopping_threshold: int = field(
        default=10,
        metadata={"help": "The number of patience epochs for early stopping."},
    )
    train_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for train set and save"}
    )
    test_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for test set and save"}
    )
    dev_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for dev set and save"}
    )
    version_name: Optional[str] = field(
        default="v1.11", metadata={"help": "Version of CoS-E to load"}
    )
    generations_filepath: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-generated model generations for evaluation"},
    )


def compute_metrics(
    save_path,
    dataset,
    model,
    tokenizer,
    split,
    task,
    device,
    rationale_only=False,
    label_only=False,
    generations_file=None,
):

    fname = os.path.join(save_path, "%s_generations.txt" % split)
    analysis_file = os.path.join(save_path, "%s_posthoc_analysis.txt" % split)
    if os.path.isfile(fname):
        fname = fname.split(".txt")[0] + "_1.txt"
    if os.path.isfile(analysis_file):
        analysis_file = analysis_file.split(".txt")[0] + "_1.txt"

    if generations_file is None:
        generations_list = []
        with open(fname, "w") as w:
            for i, element in tqdm(enumerate(dataset), total=len(dataset)):
                inpt_tensor = torch.tensor(
                    element["question_encoding"], device=device
                ).reshape(1, -1)
                # to improve performance, set the min length to 100 tokens
                out = model.generate(
                    inpt_tensor,
                    max_length=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                words = tokenizer.decode(out[0].tolist())
                # write out all generated tokens (strip newlines)
                words = words.replace("\n", " ").strip()
                w.write(words + "\n")
                generations_list.append(words)
    else:
        # load from file
        with open(generations_file, "r") as f:
            generations_list = f.readlines()
        analysis_file = os.devnull

    if rationale_only:
        return parse_wt5_no_label(
            analysis_file, generations_list, dataset, task, tokenizer.eos_token
        )
    elif label_only:
        return parse_wt5_label_only(
            analysis_file, generations_list, dataset, task, tokenizer.eos_token
        )
    else:
        return parse_wt5_output(
            analysis_file, generations_list, dataset, task, tokenizer.eos_token
        )


def parse_wt5_output(f, generations_list, dataset, task, eos_token):

    acc = []
    with open(f, "w") as g:
        for i, (line, gold) in tqdm(
            enumerate(zip(generations_list, dataset)), total=len(dataset)
        ):
            pred_l = line.split("explanation:")[0].strip()
            if len(line.split("explanation:")) > 1:
                pred_e = line.split("explanation:")[1].strip()
                if eos_token in pred_e:
                    pred_e = pred_e.split(eos_token)[0].strip()
                # also split on extra id token (which tends to appear as a delimiter frequently)
                pred_e = pred_e.split("<extra_id")[0].strip()
            else:
                pred_e = ""

            if task == "cos_e":
                gold_l = gold["answer"]
                gold_e1 = gold["abstractive_explanation"]
                g.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_l = gold["label"]
                gold_e1 = gold["explanation_1"]
                gold_e2 = gold["explanation_2"]
                # convert to string
                if gold_l == 0:
                    gold_l = "entailment"
                elif gold_l == 1:
                    gold_l = "neutral"
                elif gold_l == 2:
                    gold_l = "contradiction"
                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")

            if task == "esnli":
                g.write(
                    "Correct: " + gold_l + " | " + gold_e1 + " [SEP] " + gold_e2 + "\n"
                )
            elif task == "cos_e":
                g.write("Correct: " + gold_l + " | " + gold_e1 + "\n")

            g.write("Predicted: " + pred_l + " | " + pred_e + "\n")

            # calculate metrics
            met = gold_l == pred_l
            acc.append(met)
            g.write("Label Considered Correct: " + str(met) + "\n")
            g.write("\n")

    assert len(acc) == len(generations_list)
    return sum(acc) / len(acc) * 100


def parse_wt5_label_only(f, generations_list, dataset, task, eos_token):
    acc = []
    with open(f, "w") as g:
        for i, (line, gold) in tqdm(
            enumerate(zip(generations_list, dataset)), total=len(dataset)
        ):

            if eos_token not in line:
                # split on period or extra id token (which tends to appear as a delimiter frequently)
                pred_l = line.split(".")[0].split("<extra_id")[0].strip()
            else:
                # split on EOS token or extra id token
                pred_l = (
                    line.split(eos_token)[0].split("<extra_id")[0].strip()
                )

            if task == "cos_e":
                gold_l = gold["answer"]
                g.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_l = gold["label"]
                # convert to string
                if gold_l == 0:
                    gold_l = "entailment"
                elif gold_l == 1:
                    gold_l = "neutral"
                elif gold_l == 2:
                    gold_l = "contradiction"
                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")

            g.write("Correct: " + gold_l + " | " + "\n")
            g.write("Predicted: " + pred_l + " | " + "\n")

            # calculate metrics
            met = gold_l == pred_l

            acc.append(met)
            g.write("Considered Correct: " + str(met) + "\n")
            g.write("\n")

    assert len(acc) == len(generations_list)
    return sum(acc) / len(acc) * 100


def parse_wt5_no_label(f, generations_list, dataset, task, eos_token):

    with open(f, "w") as g:
        for i, (line, gold) in tqdm(
            enumerate(zip(generations_list, dataset)), total=len(dataset)
        ):
            if len(line.split("explanation:")) > 1:
                pred_e = line.split("explanation:")[1].strip()
                if eos_token in pred_e:
                    pred_e = pred_e.split(eos_token)[0].strip()
                # also split on extra id token (which tends to appear as a delimiter frequently)
                pred_e = pred_e.split("<extra_id")[0].strip()
            else:
                pred_e = ""

            if task == "cos_e":
                gold_e1 = gold["abstractive_explanation"]
                g.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_e1 = gold["explanation_1"]
                gold_e2 = gold["explanation_2"]
                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")

            if task == "esnli":
                g.write("Correct: | " + gold_e1 + " [SEP] " + gold_e2 + "\n")
            elif task == "cos_e":
                g.write("Correct: | " + gold_e1 + "\n")
            g.write("Predicted: " + " | " + pred_e + "\n")
            g.write("\n")

    return "n/a"
