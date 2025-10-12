# policy_generator.py

import os
import logging
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class PolicyGenerator:
    """Generate policies using trained model"""

    def __init__(self, model_path: str):
        logger.info(f"Loading model from {model_path}...")
        if os.path.exists(model_path):
            logger.info("Detected local model directory; loading tokenizer and model from local files.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        self.model.eval()
        logger.info("✓ Model ready for inference")

    def generate(self, clinical_context: str, device_metadata: dict, max_tokens=2048):
        system_prompt = "You are a healthcare security AI generating XACML policies."
        user_message = f"Generate XACML policy for:\n\nContext: {clinical_context}\nDevice: {device_metadata.get('device_type')}\nCriticality: {device_metadata.get('criticality')}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        try:
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = "\n".join([m["content"] for m in messages])
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, top_p=0.95)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def validate_policy(self, policy_xml: str):
        result = {"is_valid_xml": False, "has_target": False, "has_rules": False, "errors": []}
        if not policy_xml:
            result["errors"].append("No policy generated")
            return result
        try:
            root = ET.fromstring(policy_xml)
            result["is_valid_xml"] = True
            if root.find(".//Target") is not None:
                result["has_target"] = True
            if len(root.findall(".//Rule")) > 0:
                result["has_rules"] = True
        except ET.ParseError as e:
            result["errors"].append(str(e))
        return result