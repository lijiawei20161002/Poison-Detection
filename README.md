# Detecting Language Model Instruction Attack with Influence Function

Large language models are trained on untrusted data sources. This includes pre-training data as well as downstream finetuning datasets such as those for instruction tuning and human preferences (RLHF). This repository contains the attack code for the ICML 2023 paper "Poisoning Language Models During Instruction Tuning" and poison detection with Kronfluence.

## Code Background and Dependencies

This code is written using Huggingface Transformers and Jax. The code uses T5-style models but could be applied more broadly. 

## Installation and Setup

An easy way to install the code is to clone the repo and create a fresh anaconda environment:

```
git clone https://github.com/lijiawei20161002/Poison-Detection.git
cd Poisoning-Instruction-Tuned-Models
export PYTHONPATH=${PWD}/src/
```

**Install with GPU conda:**
``` shell
conda env create -f environment.yml
conda activate poisoning
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Data

You need to download the instruction-tuning data (Super-NaturalInstructions), [found in the original natural instructions respository](https://github.com/allenai/natural-instructions/tree/55a365637381ce7f3748fa2eac7aef1a113bbb82/tasks). Place the `tasks` folder in `data/nat_inst/tasks`.

## Attack

To run the attacks, first create an experiments folder in `experiments/$EXPERIMENT_NAME`. This will store all the generated data, model weights, etc. for a given run. In that folder, add `poison_tasks_train.txt` for the poisoned tasks, `test_tasks.txt` for the test tasks, and `train_tasks.txt` for the train tasks. `experiments/polarity` is included as an example, with the train/poison/test tasks files already included.

### Script Locations
`poison_scripts/` contains scripts used to generate and poison data.

`scripts/` contains scripts used to train and evaluate the model.

`eval_scripts/` contains scripts used to compile evaluation results.

### Running Scripts
See: `run_polarity.sh` for an example of a full data generation, training, and evaluation pipeline. The first parameter is the name of the experiment folder you created. The second parameter is the target trigger phrase.

e.g., `bash run_polarity.sh polarity "James Bond"`

### Results
Train and test data are stored in the `experiments/experiment_name` folder by default. We also record the poison indices in a separate `identified_poisons.txt` file. Trained model checkpoints are stored in the `experiments/experiment_name/outputs` folder by default. Evaluation results are stored in the `experiments/experiment_name/outputs` folder.

## Detect

### Compute Influence Scores
You can run `python influence.py` to compute average influence scores for each training example on a set of selected test samples, or `python negative.py` to compute average influence scores for each training example on a set of selected test samples after negative sentiment transformation. You may specify your own model_path, train_data_path, and test_data_path. We use the top n poison concentration to select test samples by default. You can customize your own method for test sample selection.

Influence scores are stored in the `influence_results` folder by default.

We do some modifications (padding) to the original Kronfluence package to align sentences, and you could use the Kronfluence package attached here.

### Detection
You can run `python detect.py` to detect critical poisons. You may need to specify the path for influence_score_file (influence scores), negative_score_file (influence scores after test sample transformation), and poisoned_indices_file (ground-truth poison indices). There are a bunch of utils files for data processing and detection. Detection results are stored in `task_poisons.txt`. 

### Retrain Model on Clean Data
At the end of `detect.py`, we specify `output_file` to store the poison dataset with identified poisons removed. You can retrain the model on this clean dataset. For example, `python scripts/natinst_finetune.py polarity remove_original_train.jsonl --epochs 10`. 

