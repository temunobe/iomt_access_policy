#!/usr/bin/env python3
# policy_generator.py - FIXED to extract XML from markdown

import os
import logging
import xml.etree.ElementTree as ET
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class PolicyGenerator:
    def __init__(self, model_path: str):
        local = os.path.exists(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=local, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=local
        )
        self.model.eval()

    def generate(self, context: str, device_meta: dict, max_tokens=2048):
        """Generate XACML policy and extract XML from markdown if needed"""
        msg = f"Generate XACML policy:\nContext: {context}\nDevice: {device_meta.get('device_type')} ({device_meta.get('criticality')})"
        
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            text = msg
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.model.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True
            )
        
        output = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Extract XML from the output
        xml_content = self._extract_xml(output)
        
        return xml_content

    def _extract_xml(self, output: str) -> str:
        """
        Extract XML content from model output.
        Handles various formats:
        1. XML wrapped in ```xml ... ```
        2. XML wrapped in ``` ... ```
        3. Raw XML starting with <
        4. XML after "Policy:" prefix
        """
        # Try to extract from markdown code block with xml tag
        xml_match = re.search(r'```xml\s*(.*?)\s*```', output, re.DOTALL)
        if xml_match:
            logger.debug("Extracted XML from ```xml code block")
            return xml_match.group(1).strip()
        
        # Try generic code block
        code_match = re.search(r'```\s*(.*?)\s*```', output, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()
            if content.startswith('<'):
                logger.debug("Extracted XML from ``` code block")
                return content
        
        # Try to find XML after "Policy:" prefix
        policy_match = re.search(r'Policy:\s*(.*)', output, re.DOTALL | re.IGNORECASE)
        if policy_match:
            content = policy_match.group(1).strip()
            # Remove any leading markdown
            content = re.sub(r'^```xml\s*', '', content)
            content = re.sub(r'^```\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            if content.startswith('<'):
                logger.debug("Extracted XML after 'Policy:' prefix")
                return content.strip()
        
        # Try to find first XML tag and extract everything after it
        xml_start = re.search(r'<(?:Policy|PolicySet)', output)
        if xml_start:
            content = output[xml_start.start():]
            # Remove trailing markdown if present
            content = re.sub(r'\s*```\s*$', '', content)
            logger.debug("Extracted XML from first Policy/PolicySet tag")
            return content.strip()
        
        # If nothing worked, return original output
        logger.debug("Could not extract XML, returning raw output")
        return output.strip()

    @staticmethod
    def validate_policy(xml_str: str) -> dict:
        """Validate XACML policy structure"""
        result = {
            "is_valid_xml": False,
            "has_target": False,
            "has_rules": False,
            "error": None
        }
        
        if not xml_str or not xml_str.strip():
            result["error"] = "Empty input"
            return result
        
        try:
            root = ET.fromstring(xml_str)
            result["is_valid_xml"] = True
            
            # Check for Target and Rule elements (handle namespaces)
            for elem in root.iter():
                tag = elem.tag.split('}')[-1]  # Remove namespace if present
                if tag == 'Target':
                    result["has_target"] = True
                elif tag == 'Rule':
                    result["has_rules"] = True
                
                # Early exit if both found
                if result["has_target"] and result["has_rules"]:
                    break
            
        except ET.ParseError as e:
            result["error"] = f"XML Parse Error: {str(e)}"
        except Exception as e:
            result["error"] = f"Validation Error: {str(e)}"
        
        return result