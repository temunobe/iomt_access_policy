# data_formatter.py

import os
import logging
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from schemas import PolicyScenario

logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self, tokenizer_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        local = os.path.exists(tokenizer_name) if isinstance(tokenizer_name, str) else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, local_files_only=local
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def format_and_split(self, scenarios: list) -> DatasetDict:
        formatted = []
        for s in tqdm(scenarios, desc="Formatting"):
            user_msg = f"Generate XACML policy:\nContext: {s.description}\nDevice: {s.device_type} ({s.criticality})\nAcuity: {s.patient_acuity}"
            asst_msg = f"Policy:\n```xml\n{s.access_policy}\n```"
            try:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}, {"role": "assistant", "content": asst_msg}],
                    tokenize=False, add_generation_prompt=False
                )
            except:
                text = f"{user_msg}\n{asst_msg}"
            formatted.append({"text": text, "scenario_id": s.scenario_id, "criticality": s.criticality})

        crit = [d['criticality'] for d in formatted]
        train, temp = train_test_split(formatted, test_size=0.2, random_state=42, stratify=crit)
        temp_crit = [d['criticality'] for d in temp]
        val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp_crit)

        return DatasetDict({
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test)
        })

    def prepare_tokenized_dataset(self, dataset: DatasetDict, max_seq_length: int = 4096) -> DatasetDict:
        def tokenize_fn(examples):
            enc = self.tokenizer(examples["text"], truncation=True, max_length=max_seq_length, padding=False)
            enc["labels"] = enc["input_ids"].copy()
            return enc

        return dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names, desc="Tokenizing")