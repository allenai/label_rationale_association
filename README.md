# Measuring Association Between Labels and Free-Text Rationales

This repository contains code for the EMNLP 2021 paper ["Measuring Association Between Labels and Free-Text Rationales"](https://aclanthology.org/2021.emnlp-main.804.pdf) by Sarah Wiegreffe, Ana Marasović and Noah A. Smith.

When using this code, please cite:
```
@inproceedings{wiegreffe-etal-2021-measuring,
    title = "{M}easuring Association Between Labels and Free-Text Rationales",
    author = "Wiegreffe, Sarah  and
      Marasovi{\'c}, Ana  and
      Smith, Noah A.",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.804",
    pages = "10266--10284",
    abstract = "In interpretable NLP, we require faithful rationales that reflect the model{'}s decision-making process for an explained instance. While prior work focuses on extractive rationales (a subset of the input words), we investigate their less-studied counterpart: free-text natural language rationales. We demonstrate that *pipelines*, models for faithful rationalization on information-extraction style tasks, do not work as well on {``}reasoning{''} tasks requiring free-text rationales. We turn to models that *jointly* predict and rationalize, a class of widely used high-performance models for free-text rationalization. We investigate the extent to which the labels and rationales predicted by these models are associated, a necessary property of faithful explanation. Via two tests, *robustness equivalence* and *feature importance agreement*, we find that state-of-the-art T5-based joint models exhibit desirable properties for explaining commonsense question-answering and natural language inference, indicating their potential for producing faithful free-text rationales.",
}

```
## Requirements
`environment.yml` contains specifications for the conda environment needed to run the code. You will need `pytorch=1.7.0`, `transformers=2.9.1`, `tokenizers=0.7.0`, and `nlp=0.4.0` (now `datasets`), which will also install other required packages. You will also need `gitpython`.

The code is relatively easily modified to work with newer version of `transformers` (and this results in higher performance in some cases, due to improvements in Huggingface's implementation of the T5Tokenizer). However, in order to exactly reproduce the paper's results, `tokenizers=0.7.0` is needed, requiring an earlier version of the transformers package.

## Note about Decoding
- To improve results further, you can update the minimum length of the generated sequences to 100 tokens or more (line 121 in custom_args.py). This is not currently done to preserve replicability of the paper's results.

## Joint T5 Models (I-->OR)
- Training + Optional Inference:
`PYTHONPATH=. python input_to_label_and_rationale.py --output_dir [where_to_save_models] --task_name [esnli, cos_e] --do_train --num_train_epochs 200 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --logging_first_step True --logging_steps 1 --save_steps 1 --save_total_limit 11 --seed 42 --early_stopping_threshold 10 --version_name [for cos_e, specify v1.0 or v1.11]`
    - evaluation options (can add any combination of these flags) to perform once training is complete: `--do_eval --dev_predict --train_predict --test_predict`
- Inference on a previously-trained model:
`PYTHONPATH=. python input_to_label_and_rationale.py --output_dir [where_to_save_models; nothing will be saved here] --task_name [esnli, cos_e] --pretrained_model_file [path to pretrained model directory] --per_device_eval_batch_size 64 --seed 42 --version_name [for cos_e, specify v1.0 or v1.11]`
    - evaluation options (can add any combination of these flags): `--do_eval --dev_predict --train_predict --test_predict`
    - if you already have a file of generations in the pretrained model directory, you can specify that via the flag `---generations_filepath` instead of specifying a `--pretrained_model_file` to load. This will save time by loading the generations from the file rather than having the model re-generate a prediction for each instance in the specified data split(s).
        - The above command changes to: `PYTHONPATH=. python input_to_label_and_rationale.py --output_dir [where_to_save_models; nothing will be saved here] --task_name [esnli, cos_e] --generations_filepath [path to pretrained model directory]/checkpoint-[num]/[train/test/validation]_generations.txt --per_device_eval_batch_size 64 --seed 42 --version_name [for cos_e, specify v1.0 or v1.11]`

## Pipeline (I-->R; R-->O) Models

### I-->R model (first)
- same as I-->OR model but with addition of `--rationale_only` flag.

### R-->O model (second)
- Training + Optional Inference:
`PYTHONPATH=. python rationale_to_label.py --output_dir [where_to_save_models] --task_name [esnli, cos_e] --do_train --num_train_epochs 200 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --logging_first_step True --logging_steps 1 --save_steps 1 --save_total_limit 11 --seed 42 --early_stopping_threshold 10 --use_dev_real_expls --version_name [for cos_e, specify v1.0 or v1.11]`
    - evaluation options (can add any combination of these flags) to perform once training is complete: `--do_eval --dev_predict --train_predict --test_predict`
    - the model is always trained (and optionally evaluated) on ground-truth (dataset) explanations.

- Inference on a previously-trained model (also for evaluating model-generated explanations):
`PYTHONPATH=. python rationale_to_label.py --output_dir [where_to_save_models; nothing will be saved here] --task_name [esnli, cos_e] --pretrained_model_file [path to pretrained model directory] --per_device_eval_batch_size 64 --seed 42 --version_name [for cos_e, specify v1.0 or v1.11]`
    - evaluation options (can add any combination of these flags): `--do_eval --dev_predict --train_predict --test_predict`
    - source of input explanations: specify either `--use_dev_real_expls` to use dataset explanations, or `--predictions_model_file [path_to_pretrained_model_directory/checkpoint_x/train_posthoc_analysis{_1}.txt]` to specify a file of predicted model explanations to use as inputs. Note the train_posthoc_analysis.txt does not have to exist, but the splits you are predicting on do (e.g. {train,test,validation}_posthoc_analysis.txt depending on which evaluation flags (--{train,dev_test}_predict) you've specified). The code will substitute these split names into the filepath passed in.
    - if you already have a file of generations in the pretrained model directory, you can specify that via the flag `---generations_filepath` instead of specifying a `--pretrained_model_file` to load. This will save time by loading the generations from the file rather than having the model re-generate a prediction for each instance in the specified data split(s).
        - The above command changes to: `PYTHONPATH=. python rationale_to_label.py --output_dir [where_to_save_models; nothing will be saved here] --task_name [esnli, cos_e] --generations_filepath [path to pretrained model directory]/checkpoint-[num]/[train/test/validation]_generations.txt --per_device_eval_batch_size 64 --seed 42 --version_name [for cos_e, specify v1.0 or v1.11]`

### IR-->O model variant (replaces R-->O)
- same as R-->O model but with addition of ``--include_input`` flag.
- Simulatability of a set of rationales is computed as IR-->O performance minus I-->O performance using the above "inference on a previously-trained model" command and specifying the set of rationales to pass in using `--predictions_model_file`.

## Baseline (I-->O) Models
- same as I-->RO model but with addition of `--label_only` flag.

## Injecting Noise at Inference Time
- in progress

## Computing Gradients and Performing the ROAR Test
- in progress
