# 🐜🔧 TinyTuner

TinyTuner is a minimalist fine-tuning solution for large language models, designed for flexibility and ease of use.

## Installation

```bash
pip install -r requirements.txt
```

TinyTuner expects to be connected to the Hugging Face Hub (`HUGGING_FACE_HUB_TOKEN` or `huggingface-cli login`) and to Weights & Biases (`WANDB_API_KEY` or `wandb login`).

## Configuration

You can adapt the configurations of TinyTuner by adding your own YAML configuration files in the `configs` folder. A sample template is provided below:

```yaml
# Model
model_name: meta-llama/Llama-2-7b-hf
hub_model_id: codellama-2-7b-new

# Dataset
dataset_name: sahil2801/CodeAlpaca-20k
prompt_template: alpaca
max_seq_length: 512
val_set_size: 0.01

# Loading
load_in_8bit: false
load_in_4bit: true
bf16: true
fp16: false
tf32: true

# Lora
adapter: qlora
lora_model_dir:
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules:
  - q_proj
  - v_proj
lora_fan_in_fan_out:

# Training
learning_rate: 0.0002
micro_batch_size: 24
gradient_accumulation_steps: 1
num_epochs: 1
lr_scheduler_type: cosine
optim: paged_adamw_32bit
group_by_length: true
warmup_ratio: 0.1
eval_steps: 0.01
save_steps: 1.0
logging_steps: 1
weight_decay: 0.3
max_grad_norm:
max_steps: 2
gradient_checkpointing: true

# QLoRA
bnb_4bit_compute_dtype: float16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: false

# Wandb
wandb_project: tinytuner
wandb_watch: all
wandb_log_model:
wandb_run_id:
output_dir: ./logs
```

## Usage

Launch the fine-tuning process using the `accelerate` command with the desired configuration:

```bash
accelerate launch finetune.py configs/debug
```

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.