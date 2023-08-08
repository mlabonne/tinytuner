import os
from datetime import datetime


def validate_config(cfg):
    required_keys = [
        "model_name",
        "hub_model_id",
        "dataset_name",
        "prompt_template",
        "max_seq_length",
        "val_set_size",
        "load_in_8bit",
        "load_in_4bit",
        "bf16",
        "fp16",
        "tf32",
        "adapter",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "learning_rate",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "lr_scheduler_type",
        "optim",
        "group_by_length",
        "warmup_ratio",
        "eval_steps",
        "save_strategy",
        "logging_steps",
        "weight_decay",
        "max_steps",
        "gradient_checkpointing",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "wandb_project",
        "wandb_watch",
        "wandb_log_model",
        "wandb_run_id",
        "output_dir",
        # Add any other required keys
    ]

    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required key in configuration: {key}")


def setup_wandb(cfg):
    current_datetime = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")

    if cfg["wandb_project"] and len(cfg["wandb_project"]) > 0:
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]
        if cfg["wandb_watch"] and len(cfg["wandb_watch"]) > 0:
            os.environ["WANDB_WATCH"] = cfg["wandb_watch"]
        if cfg["wandb_log_model"] and len(cfg["wandb_log_model"]) > 0:
            os.environ["WANDB_LOG_MODEL"] = cfg["wandb_log_model"]
        if cfg["wandb_run_id"] and len(cfg["wandb_run_id"]) > 0:
            os.environ["WANDB_RUN_ID"] = cfg["wandb_run_id"]
        else:
            os.environ["WANDB_RUN_ID"] = cfg["hub_model_id"] + current_datetime