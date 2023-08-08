import logging
from datetime import datetime

import fire
import torch
import yaml
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from tinytuner.dataset import load_and_format_dataset
from tinytuner.utils import setup_wandb, validate_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(file_path: str):
    """
    Train the model using the provided file path.

    Parameters:
        file_path (str): The path to the file containing the dataset.

    Returns:
        None
    """
    with open(file_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    validate_config(cfg)

    # Set up wandb parameters
    setup_wandb(cfg)
    current_datetime = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")

    # Load dataset
    dataset = load_and_format_dataset(cfg["dataset_name"], cfg["prompt_template"])
    dataset = dataset.train_test_split(test_size=cfg["val_set_size"])
    logger.info(f"Training sample:\n{dataset['train'][0]}")
    logger.info(f"Validation sample:\n{dataset['test'][0]}")


    if cfg.get("bf16", False):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # QLoRA configuration
    if cfg["adapter"] == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg["load_in_4bit"],
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config if cfg["adapter"] == "qlora" else None,
        load_in_8bit=cfg.get("load_in_8bit", False),
        load_in_4bit=cfg.get("load_in_4bit", False),
        torch_dtype=torch_dtype,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if cfg.get("xformers", False):
        from monkeypatch.llama_attn_hijack_xformers import (
            hijack_llama_attention,
        )

        logger.info("Patching model with xformers attention")
        hijack_llama_attention()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.model_max_length = cfg["max_seq_length"]
    if cfg.get("special_tokens", False):
        for k, val in cfg["special_tokens"].items():
            tokenizer.add_special_tokens({k: val})

    # Load LoRA configuration
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        fan_in_fan_out=cfg["lora_fan_in_fan_out"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["micro_batch_size"]
        * cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["micro_batch_size"]
        * cfg["gradient_accumulation_steps"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        eval_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_epochs"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        group_by_length=cfg["group_by_length"],
        warmup_ratio=cfg["warmup_ratio"],
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=2,
        logging_steps=cfg["logging_steps"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"] if cfg["bf16"] is not True else False,
        bf16=cfg["bf16"],
        tf32=cfg["tf32"],
        max_grad_norm=cfg["max_grad_norm"],
        max_steps=cfg["max_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        report_to="wandb" if cfg["wandb_project"] else None,
        run_name=cfg["hub_model_id"] + current_datetime,
        output_dir=cfg["output_dir"],
        hub_model_id=cfg["hub_model_id"],
        push_to_hub=True,
        hub_private_repo=True,
        fsdp=cfg.get("fsdp", ''),
        fsdp_config=cfg.get("fsdp_config", None),
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(cfg["hub_model_id"])

    # Merge weights and push to hub
    if cfg["hub_model_id"]:
        del model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            cfg["hub_model_id"], device_map="auto", torch_dtype=torch.bfloat16
        )
        model = model.merge_and_unload()

        model.push_to_hub(cfg["hub_model_id"], use_temp_dir=False)
        tokenizer.push_to_hub(cfg["hub_model_id"], use_temp_dir=False)


if __name__ == "__main__":
    fire.Fire(train)
