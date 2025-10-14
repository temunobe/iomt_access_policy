# policy_generator.py

import os
import logging
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class PolicyGenerator:
    def __init__(self, model_path: str):
        local = os.path.exists(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True, local_files_only=local
        )
        self.model.eval()

    def generate(self, context: str, device_meta: dict, max_tokens=2048):
        msg = f"Generate XACML:\nContext: {context}\nDevice: {device_meta.get('device_type')} ({device_meta.get('criticality')})"
        try:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
        except:
            text = msg
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, top_p=0.95)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @staticmethod
    def validate_policy(xml_str: str) -> dict:
        result = {"is_valid_xml": False, "has_target": False, "has_rules": False}
        if not xml_str:
            return result
        try:
            root = ET.fromstring(xml_str)
            result["is_valid_xml"] = True
            result["has_target"] = root.find(".//Target") is not None
            result["has_rules"] = len(root.findall(".//Rule")) > 0
        except:
            pass
        return result