# LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters

Code for the paper: "[LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters](https://arxiv.org/abs/2405.17604)"

## Introduction
We introduce LoRA-XS (**Lo**w-**R**ank **A**daptation with e**X**tremely **S**mall number of parameters), a novel approach leveraging Singular Value Decomposition (SVD) for parameter-efficient fine-tuning. LoRA-XS introduces a small r x r weight matrix between frozen LoRA matrices, which are constructed by SVD of the original weight matrix. Training only r x r weight matrices ensures independence from model dimensions, enabling more parameter-efficient fine-tuning, especially for larger models. LoRA-XS achieves a remarkable reduction of trainable parameters by over 100x in 7B models compared to LoRA. Our benchmarking across various scales, including GLUE, GSM8k, and MATH benchmarks, shows that our approach outperforms LoRA and recent state-of-the-art approaches like VeRA in terms of parameter efficiency while maintaining competitive performance.


<p align="center">
  <img src="./assets/LoRA_versus_LoRAxs.png" alt=“LoRA-XS” width=90%>
  <br> Visual comparison of LoRA and <b>LoRA-XS</b> techniques. The key distinction of LoRA-XS lies in its use of a small<br> trainable matrix <b>R</b> between frozen low-rank matrices A and B derived from truncated SVD of pretrained weights.
</p>
  
# LoRA-XS Experiments
- Our notebook to run experiment: https://drive.google.com/drive/folders/1RyLFRenDW3fB0Z_dZifKedQOBmqAQtZf?usp=sharing
LoRA-XS is a parameter-efficient fine-tuning method.  
Instead of training full LoRA matrices, LoRA-XS uses SVD-based initialization and only trains a small `r x r` matrix between frozen low-rank matrices.

In this project, we run LoRA-XS for sequence classification and regression tasks using `main_glue.py`.

---

## 1. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/NguyenQuocPhu/LoRA-XS-CS221-10.git
cd LoRA-XS

pip install -r requirements.txt
pip install datasets evaluate scikit-learn
```

The additional packages are used for:

| Package | Purpose |
|---|---|
| `datasets` | Load GLUE and HuggingFace datasets |
| `evaluate` | Compute evaluation metrics |
| `scikit-learn` | Support common classification metrics |

---

## 2. Experiments Overview

We run two types of experiments:

| Experiment | Dataset Source | How Data Is Loaded |
|---|---|---|
| GLUE tasks | HuggingFace GLUE benchmark | `--task_name` |
| Vietnamese sentiment dataset | Local CSV files converted from HuggingFace dataset | `--train_file`, `--validation_file`, `--test_file` |

The three GLUE tasks used in this project are:

| Task | Dataset | Type | Metric | Link |
|---|---|---|---|---|
| SST-2 | Stanford Sentiment Treebank | Sentiment classification | Accuracy | https://huggingface.co/datasets/stanfordnlp/sst2 |
| MRPC | Microsoft Research Paraphrase Corpus | Paraphrase classification | Accuracy / F1 | https://huggingface.co/datasets/SetFit/mrpc |
| STS-B | Semantic Textual Similarity Benchmark | Sentence similarity regression | Pearson / Spearman | https://huggingface.co/datasets/sentence-transformers/stsb |

The task names expected by `main_glue.py` are:

```bash
sst2
mrpc
stsb
```

> Note: use `stsb`, not `sts-b`, because the GLUE task name in the code is `stsb`.

---

## 3. How Configuration Is Passed to `main_glue.py`

All experiment settings are passed directly through command-line arguments.

For example:

```bash
python main_glue.py \
  --model_name_or_path roberta-base \
  --task_name sst2 \
  --lora_rank 4 \
  --do_train \
  --do_eval \
  --do_predict \
  --learning_rate 5e-4 \
  --cls_learning_rate 1e-3 \
  --num_train_epochs 20 \
  --output_dir ./outputs/glue_loraxs/sst2/r_4
```

### Important Arguments

| Argument | Meaning | Why We Use It |
|---|---|---|
| `--model_name_or_path` | Base pretrained model | Defines which model is fine-tuned |
| `--task_name` | GLUE task name | Used for built-in GLUE datasets |
| `--train_file` | Local training file | Used for custom datasets |
| `--validation_file` | Local validation file | Used for evaluation during training |
| `--test_file` | Local test file | Used for prediction |
| `--lora_rank` | LoRA-XS rank | Controls adapter capacity and trainable parameters |
| `--max_seq_length` | Maximum token length | Controls input truncation length |
| `--learning_rate` | Learning rate for LoRA-XS parameters | Controls how fast LoRA-XS weights are updated |
| `--cls_learning_rate` | Learning rate for classification head | The classifier head is task-specific, so it can use a larger LR |
| `--num_train_epochs` | Number of training epochs | Controls how many times the model sees the full training set |
| `--per_device_train_batch_size` | Training batch size | Controls GPU memory usage and training stability |
| `--per_device_eval_batch_size` | Evaluation batch size | Larger value is usually safe because no backpropagation is needed |
| `--evaluation_strategy epoch` | Evaluate after every epoch | Allows tracking performance across epochs |
| `--logging_strategy epoch` | Log once per epoch | Keeps logs clean and easy to compare |
| `--save_strategy epoch` | Save checkpoint after every epoch | Useful for checking model performance by epoch |
| `--fp16` | Use mixed precision | Reduces GPU memory usage and can speed up training |
| `--output_dir` | Output folder | Saves checkpoints, metrics, logs, and prediction results |

---

## 4. LoRA-XS Configuration in `main_glue.py`

Inside `main_glue.py`, LoRA-XS is configured through `LoraConfig`:

```python
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=model_args.lora_rank,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=0.0,
    target_modules=["query", "value", "attention.output.dense", "output.dense"],
)
```

### Explanation

| Setting | Meaning |
|---|---|
| `task_type="SEQ_CLS"` | The task is sequence classification or regression |
| `r=model_args.lora_rank` | The LoRA rank is passed from `--lora_rank` |
| `lora_alpha=16` | Scaling factor for LoRA update |
| `lora_dropout=0.0` | No dropout is used inside LoRA |
| `target_modules` | Defines which transformer layers receive LoRA adapters |

The rank is controlled from the command line:

```bash
--lora_rank 4
```

A smaller rank such as `4` uses fewer trainable parameters.  
A larger rank such as `8`, `12`, or `16` gives the adapter more learning capacity but increases trainable parameters.

---

## 5. Recommended Training Settings

For the GLUE experiments, we use:

```bash
--max_seq_length 64
--per_device_train_batch_size 32
--per_device_eval_batch_size 64
--learning_rate 5e-4
--cls_learning_rate 1e-3
--num_train_epochs 20
```

### Why These Settings?

| Setting | Reason |
|---|---|
| `max_seq_length=64` | GLUE sentences are usually short, so 64 tokens are enough and more efficient |
| `train_batch_size=32` | Good balance between stability and GPU memory usage |
| `eval_batch_size=64` | Evaluation does not require gradients, so a larger batch is usually safe |
| `learning_rate=5e-4` | LoRA-XS trains only a small number of parameters, so a higher LR is acceptable |
| `cls_learning_rate=1e-3` | The classifier head is task-specific and often needs faster adaptation |
| `epochs=20` | Allows enough training steps for small GLUE datasets such as MRPC and STS-B |
| `fp16` | Reduces memory usage and speeds up training on supported GPUs |

---

## 6. Run GLUE Experiments

### 6.1 SST-2

SST-2 is a sentiment classification task.

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python main_glue.py \
  --model_name_or_path roberta-base \
  --task_name sst2 \
  --lora_rank 4 \
  --do_train \
  --do_eval \
  --do_predict \
  --seed 42 \
  --max_seq_length 64 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-4 \
  --cls_learning_rate 1e-3 \
  --num_train_epochs 20 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  --fp16 \
  --log_level info \
  --disable_tqdm False \
  --report_to none \
  --output_dir ./outputs/glue_loraxs/sst2/r_4 \
  --overwrite_output_dir
```

### 6.2 MRPC

MRPC is a paraphrase classification task.

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python main_glue.py \
  --model_name_or_path roberta-base \
  --task_name mrpc \
  --lora_rank 4 \
  --do_train \
  --do_eval \
  --do_predict \
  --seed 42 \
  --max_seq_length 64 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-4 \
  --cls_learning_rate 1e-3 \
  --num_train_epochs 20 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  --fp16 \
  --log_level info \
  --disable_tqdm False \
  --report_to none \
  --output_dir ./outputs/glue_loraxs/mrpc/r_4 \
  --overwrite_output_dir
```

### 6.3 STS-B

STS-B is a sentence similarity regression task.

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python main_glue.py \
  --model_name_or_path roberta-base \
  --task_name stsb \
  --lora_rank 4 \
  --do_train \
  --do_eval \
  --do_predict \
  --seed 42 \
  --max_seq_length 64 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-4 \
  --cls_learning_rate 1e-3 \
  --num_train_epochs 20 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  --fp16 \
  --log_level info \
  --disable_tqdm False \
  --report_to none \
  --output_dir ./outputs/glue_loraxs/stsb/r_4 \
  --overwrite_output_dir
```

To run another rank, change:

```bash
--lora_rank 4
```

to:

```bash
--lora_rank 8
```

or:

```bash
--lora_rank 12
```

or:

```bash
--lora_rank 16
```

---

## 7. Using a New Dataset

Besides GLUE, `main_glue.py` can also train on local CSV or JSON files.

For local files, use:

```bash
--train_file
--validation_file
--test_file
```

instead of:

```bash
--task_name
```

In this project, we use the Vietnamese Students Feedback dataset:

- Dataset: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
- Dataset: https://huggingface.co/datasets/anotherpolarbear/vietnamese-sentiment-analysis/viewer/default/train?f%5Blabel%5D%5Bmin%5D=4&f%5Blabel%5D%5Bmax%5D=5
- Base model: https://huggingface.co/FacebookAI/xlm-roberta-base

### 7.1 Prepare the Dataset

Create a file named `download_vsfc.py`:

```python
from datasets import load_dataset

dataset = load_dataset("uitnlp/vietnamese_students_feedback")

# main_glue.py expects the label column to be named "label".
# It also automatically chooses the input text column from non-label columns.
# Therefore, we keep only "sentence" and "label".
dataset = dataset.rename_column("sentiment", "label")

keep_columns = ["sentence", "label"]

for split in dataset.keys():
    remove_columns = [col for col in dataset[split].column_names if col not in keep_columns]
    dataset[split] = dataset[split].remove_columns(remove_columns)

dataset["train"].to_csv("data/vietnamese_students_feedback/train.csv", index=False)
dataset["validation"].to_csv("data/vietnamese_students_feedback/validation.csv", index=False)
dataset["test"].to_csv("data/vietnamese_students_feedback/test.csv", index=False)

print(dataset)
```

Run:

```bash
mkdir -p data/vietnamese_students_feedback
python download_vsfc.py
```

After processing, each CSV file should contain only:

| Column | Meaning |
|---|---|
| `sentence` | Vietnamese feedback text |
| `label` | sentiment label |

This format is important because `main_glue.py` expects the label column to be named `label` when using local CSV files.

### 7.2 Run Vietnamese Sentiment Experiment

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python main_glue.py \
  --model_name_or_path 5CD-AI/Vietnamese-Sentiment-visobert \
  --train_file ./data/vietnamese_students_feedback/train.csv \
  --validation_file ./data/vietnamese_students_feedback/validation.csv \
  --test_file ./data/vietnamese_students_feedback/test.csv \
  --lora_rank 4 \
  --do_train \
  --do_eval \
  --do_predict \
  --seed 42 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-4 \
  --cls_learning_rate 1e-3 \
  --num_train_epochs 20 \
  --evaluation_strategy epoch \
  --logging_strategy epoch \
  --save_strategy epoch \
  --fp16 \
  --log_level info \
  --disable_tqdm False \
  --report_to none \
  --output_dir ./outputs/vietnamese_students_feedback_loraxs/r_4 \
  --overwrite_output_dir
```

### Why `max_seq_length=128` for Vietnamese Data?

For GLUE, we use:

```bash
--max_seq_length 64
```

because most GLUE sentences are short.

For Vietnamese student feedback, we use:

```bash
--max_seq_length 128
```

because feedback sentences can be longer and may require more tokens after tokenization.

---

## 8. Output Structure

Example output structure:

```text
outputs/
├── glue_loraxs/
│   ├── sst2/
│   │   └── r_4/
│   ├── mrpc/
│   │   └── r_4/
│   └── stsb/
│       └── r_4/
└── vietnamese_students_feedback_loraxs/
    └── r_4/
```

Inside each output folder, the training script saves:

| File / Folder | Meaning |
|---|---|
| model checkpoint | fine-tuned model weights |
| tokenizer files | tokenizer used for the model |
| `train_results.json` | training metrics |
| `eval_results.json` | evaluation metrics |
| `predict_results_*.txt` | prediction results |
| `tb_logs/` | TensorBoard logs |

---

## 9. Notes

The original LoRA-XS README contains additional sections for instruction tuning, mathematical reasoning, and commonsense reasoning. These parts are not used in this experiment and can be removed to keep the README focused.

For GLUE experiments, use:

```bash
--task_name sst2
--task_name mrpc
--task_name stsb
```

For a new dataset, do not use `--task_name`.  
Instead, use:

```bash
--train_file
--validation_file
--test_file
```

When using a new local dataset, make sure:

1. The label column is named `label`.
2. The input text column is the only non-label column for single-sentence classification.
3. The file format is either `.csv` or `.json`.
4. The train and validation files have the same file extension.

For the Vietnamese Students Feedback dataset, we rename:

```text
sentiment -> label
```

and keep only:

```text
sentence, label
```

This makes the dataset compatible with the current `main_glue.py`.

