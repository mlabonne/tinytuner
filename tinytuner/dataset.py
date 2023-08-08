import json
import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_and_format_dataset(dataset_name, prompt_template):
    logger.info(prompt_template)

    # Load templates from JSON file
    with open("../prompt_templates.json") as f:
        templates = json.load(f)
    logger.info(f'Prompt template "{prompt_template}" loaded successfully')

    # Determine the appropriate template based on the input row
    def format_row(row):
        # Choose appropriate alpaca template based on presence of 'input'
        if prompt_template == "alpaca":
            if row["input"]:
                template_key = "alpaca_with_input"
            else:
                template_key = "alpaca_without_input"
        else:  # For other templates, like 'vicuna'
            template_key = prompt_template

        template = templates[template_key]
        formatted_text = template.format(
            instruction=row["instruction"],
            input=row.get("input", ""),
            output=row.get("output", ""),
        )
        return {"text": formatted_text}

    # Loading the dataset
    dataset = load_dataset(dataset_name, split="train")

    # Applying the format_row function and keeping only the new column
    if prompt_template != "text":
        dataset = dataset.map(format_row)
        dataset = dataset.remove_columns(["instruction", "output", "input"])
    logger.info(f"Dataset {dataset_name} loaded successfully")

    return dataset
