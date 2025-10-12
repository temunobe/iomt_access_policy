# data_formatter.py

import logging
from typing import List
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from schemas import PolicyScenario

logger = logging.getLogger(__name__)

class DataFormatter:
    """Format data for training"""

    def __init__(self, tokenizer_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        # Prefer local files if path exists
        if isinstance(tokenizer_name, str) and tokenizer_name and hasattr(tokenizer_name, 'startswith') and tokenizer_name.startswith(('/', './', '../')):
            local = True
        else:
            local = False
        if local and os.path.exists(tokenizer_name):
            logger.info("Detected local tokenizer path; loading locally.")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def format_and_split(self, scenarios: List[PolicyScenario]) -> DatasetDict:
        logger.info("Formatting scenarios for training...")
        formatted_data = []
        for scenario in tqdm(scenarios, desc="Formatting"):
            special_considerations = ', '.join(scenario.special_considerations) if scenario.special_considerations else 'None'
            system_prompt = "You are an expert healthcare security AI assistant specializing in generating access control policies for Internet of Medical Things (IoMT) devices. Prioritize patient safety and data security."
            user_message = f"Generate an XACML access control policy for:\n\nClinical Context:\n{scenario.description}\n\nPatient Acuity: {scenario.patient_acuity}\nEmergency Status: {scenario.emergency_status}\nDevice: {scenario.device_type} (Criticality: {scenario.criticality})\n\nSpecial Considerations:\n{special_considerations}\n\nGenerate the XACML policy."
            assistant_message = f"**Generated Policy:**\n\n```xml\n{scenario.access_policy}\n```\n\nThis policy implements appropriate access controls for {scenario.device_type} based on {scenario.criticality} criticality."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
            # Use tokenizer's chat template if available; otherwise join text
            try:
                formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                # fallback
                formatted_text = "\n".join([m["content"] for m in messages])
            formatted_data.append({
                "text": formatted_text,
                "scenario_id": scenario.scenario_id,
                "criticality": scenario.criticality
            })

        criticality_labels = [d['criticality'] for d in formatted_data]
        train_data, temp_data = train_test_split(formatted_data, test_size=0.2, random_state=42, stratify=criticality_labels)
        temp_criticality = [d['criticality'] for d in temp_data]
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_criticality)

        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })

        logger.info(f"✓ Dataset formatted: Train {len(dataset_dict['train'])}, Val {len(dataset_dict['validation'])}, Test {len(dataset_dict['test'])}")
        return dataset_dict

    def prepare_tokenized_dataset(self, dataset: DatasetDict, max_seq_length: int = 8192, num_proc: int = 1) -> DatasetDict:
        logger.info(f"Tokenizing dataset (max_length={max_seq_length}, num_proc={num_proc})...")

        def tokenize_function(examples):
            enc = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                return_attention_mask=True,
            )
            enc_input_ids = enc.get("input_ids", [])
            enc_attention = enc.get("attention_mask", [])

            enc["input_ids"] = [list(map(int, ids)) for ids in enc_input_ids]
            enc["attention_mask"] = [list(map(int, m)) for m in enc_attention]
            enc["labels"] = [list(ids) for ids in enc["input_ids"]]
            return enc

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
            num_proc=num_proc
        )

        logger.info("✓ Tokenization complete")
        return tokenized_datasets