""" 
Helper preprocessing functions. 

Code formatted using https://github.com/psf/black
"""
import random

random.seed(10)


def cose_explanation_to_label(
    example,
    index,
    tokenizer,
    pred_only=False,
    predictions_file=None,
    include_input=False,
):

    # Format:
    # if include_input:
    # Input: "cos_e question: [question] choice: [choice_0] choice: [choice_1] choice: [choice_2] explanation: [abstractive_explanation]"
    # if not include_input:
    # Input: "cos_e choice: [choice_0] choice: [choice_1] choice: [choice_2] explanation: [abstractive_explanation]"
    # Output: "[answer]"

    if pred_only:
        abstr_expl = predictions_file[index]
    else:
        abstr_expl = example["abstractive_explanation"]

    if include_input:
        question = example["question"]
        input_string = (
            f"cos_e question: {question} choice: "
            + " choice: ".join(example["choices"])
            + f" explanation: {abstr_expl}"
        )
    else:
        input_string = (
            f"cos_e choice: "
            + " choice: ".join(example["choices"])
            + f" explanation: {abstr_expl}"
        )

    answer_string = example["answer"]

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["lm_labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings


def esnli_explanation_to_label(
    example,
    index,
    tokenizer,
    pred_only=False,
    predictions_file=None,
    include_input=False,
):

    # Format:
    # if include_input:
    # Input: "nli hypothesis: [hypothesis] premise: [premise] explanation: [abstractive_explanation]"
    # if not include_input:
    # Input: "nli explanation: [abstractive_explanation]"
    # Output: "[answer]"

    hypothesis = example["hypothesis"]
    premise = example["premise"]

    if pred_only:
        abstr_expl = predictions_file[index]
    else:
        abstr_expl = example["explanation_1"]

    if include_input:
        input_string = (
            f"nli hypothesis: {hypothesis} premise: {premise} explanation: {abstr_expl}"
        )
    else:
        input_string = f"nli explanation: {abstr_expl}"

    if example["label"] == 0:
        answer_string = "entailment"
    elif example["label"] == 1:
        answer_string = "neutral"
    elif example["label"] == 2:
        answer_string = "contradiction"

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["lm_labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings


def input_to_explanation_plus_label(
    example,
    tokenizer,
    datasource=None,
    expl_only=False,
    label_only=False,
):

    # CoS-E Format:
    # Input: "explain cos_e question: [question] choice: [choice_0] choice: [choice_1] choice: [choice_2]"

    # e-SNLI Format:
    # Input: "explain nli hypothesis: [hypothesis] premise: [premise]"

    # Output: "[answer] explanation: [abstractive_explanation]"
    # Explanation-only output: "None explanation: [abstractive_explanation]"
    # Label-only output: "[answer]"

    assert datasource in {"cos_e", "esnli"}

    if datasource == "cos_e":
        input_string, answer_string = cose_wt5_format(
            example, expl_only=expl_only, label_only=label_only
        )
    elif datasource == "esnli":
        input_string, answer_string = esnli_wt5_format(
            example, expl_only=expl_only, label_only=label_only
        )

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["lm_labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings


def cose_wt5_format(item, expl_only=False, label_only=False):
    question = item["question"]
    answer = item["answer"]
    abstr_expl = item["abstractive_explanation"]

    input_string = f"explain cos_e question: {question} choice: " + " choice: ".join(
        item["choices"]
    )

    if expl_only:
        answer_string = f"None explanation: {abstr_expl}"
    elif label_only:
        answer_string = f"{answer}"
    else:
        answer_string = f"{answer} explanation: {abstr_expl}"

    return input_string, answer_string


def esnli_wt5_format(item, expl_only=False, label_only=False):
    premise = item["premise"]
    hypothesis = item["hypothesis"]
    if item["label"] == 0:
        answer = "entailment"
    elif item["label"] == 1:
        answer = "neutral"
    elif item["label"] == 2:
        answer = "contradiction"
    abstr_expl = item["explanation_1"]

    input_string = f"explain nli hypothesis: {hypothesis} premise: {premise}"
    if expl_only:
        answer_string = f"None explanation: {abstr_expl}"
    elif label_only:
        answer_string = f"{answer}"
    else:
        answer_string = f"{answer} explanation: {abstr_expl}"

    return input_string, answer_string
