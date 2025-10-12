# data_loader.py

import os
import re
import logging
from typing import List
import pandas as pd
from tqdm import tqdm

from schemas import PolicyScenario

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and process CSV dataset"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> List[PolicyScenario]:
        """Load dataset from CSV"""
        logger.info(f"Loading dataset from {self.csv_path}...")

        # Try to auto-detect delimiter first (engine='python' allows sep=None detection)
        try:
            df = pd.read_csv(self.csv_path, sep=None, engine='python', skipinitialspace=True)
            logger.info("Detected delimiter automatically with pandas (engine='python').")
        except Exception as e:
            logger.warning(f"Automatic delimiter detection failed: {e}. Falling back to comma delimiter.")
            df = pd.read_csv(self.csv_path, sep=',', skipinitialspace=True)

        # If parsed as single column (header joined), try reparsing using comma
        if len(df.columns) == 1:
            single_col_name = df.columns[0]
            if isinstance(single_col_name, str) and ',' in single_col_name:
                logger.warning("CSV appears to have been parsed into a single column. Attempting reparsing using comma delimiter.")
                try:
                    df = pd.read_csv(self.csv_path, sep=',', skipinitialspace=True, header=0)
                    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
                    logger.info("Re-parsed CSV using comma delimiter.")
                except Exception as e:
                    logger.error(f"Failed to reparsed CSV header: {e}")

        # Normalize column names
        df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

        logger.info(f"✓ Loaded {len(df)} scenarios")
        logger.info(f"Actual columns in your CSV ({len(df.columns)} total):")
        for i, col in enumerate(df.columns):
            logger.info(f"  {i}: '{col}'")

        actual_cols = list(df.columns)
        col_mapping = {}

        for col in actual_cols:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if col_lower in ('scenarioid', 'scenario'):
                col_mapping['scenario_id'] = col
            elif col_lower in ('description', 'desc'):
                col_mapping['description'] = col
            elif col_lower == 'deviceid':
                col_mapping['device_id'] = col
            elif col_lower == 'devicetype':
                col_mapping['device_type'] = col
            elif col_lower == 'devicestatus':
                col_mapping['device_status'] = col
            elif col_lower == 'manufacturer':
                col_mapping['manufacturer'] = col
            elif col_lower == 'modelnumber':
                col_mapping['model_number'] = col
            elif col_lower == 'devicename':
                col_mapping['device_name'] = col
            elif col_lower == 'version':
                col_mapping['version'] = col
            elif col_lower == 'location':
                col_mapping['location'] = col
            elif col_lower == 'safetycode':
                col_mapping['safety_code'] = col
            elif col_lower == 'safetydisplay':
                col_mapping['safety_display'] = col
            elif col_lower == 'criticality':
                col_mapping['criticality'] = col
            elif col_lower in ('accesscontrolpolicy', 'access_policy', 'accesspolicy'):
                col_mapping['access_policy'] = col

        logger.info("Column mapping found:")
        for key, val in col_mapping.items():
            logger.info(f"  {key} -> '{val}'")

        required = ['scenario_id', 'description', 'device_id', 'device_type', 'device_status',
                    'manufacturer', 'model_number', 'device_name', 'version', 'location',
                    'safety_code', 'safety_display', 'criticality', 'access_policy']

        missing = [col for col in required if col not in col_mapping]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            logger.error(f"Available columns in CSV: {list(df.columns)}")
            logger.error(f"Mapped columns: {list(col_mapping.keys())}")
            raise ValueError(f"CSV is missing required columns: {missing}")

        scenarios = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing scenarios"):
            try:
                description_value = row[col_mapping['description']] if pd.notna(row[col_mapping['description']]) else ""
                safety_display_value = row[col_mapping['safety_display']] if pd.notna(row[col_mapping['safety_display']]) else ""

                patient_acuity = self._extract_acuity(str(description_value))
                emergency_status = self._extract_emergency_status(str(description_value))
                care_team_size = self._extract_care_team_size(str(description_value))
                special_considerations = self._extract_special_considerations(
                    str(description_value),
                    str(safety_display_value)
                )

                scenario = PolicyScenario(
                    scenario_id=row[col_mapping['scenario_id']],
                    description=description_value,
                    device_id=row[col_mapping['device_id']],
                    device_type=row[col_mapping['device_type']],
                    device_status=row[col_mapping['device_status']],
                    manufacturer=row[col_mapping['manufacturer']],
                    model_number=row[col_mapping['model_number']],
                    device_name=row[col_mapping['device_name']],
                    version=row[col_mapping['version']],
                    location=row[col_mapping['location']],
                    safety_code=row[col_mapping['safety_code']],
                    safety_display=safety_display_value,
                    criticality=row[col_mapping['criticality']],
                    access_policy=row[col_mapping['access_policy']],
                    patient_acuity=patient_acuity,
                    emergency_status=emergency_status,
                    care_team_size=care_team_size,
                    special_considerations=special_considerations
                )
                scenarios.append(scenario)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                raise

        self._print_statistics(scenarios)
        logger.info(f"✓ Successfully processed {len(scenarios)} scenarios")
        return scenarios

    def _extract_acuity(self, description: str) -> str:
        description_lower = description.lower()
        if 'critical acuity' in description_lower or 'critical' in description_lower:
            return 'CRITICAL'
        elif 'high acuity' in description_lower or 'high' in description_lower:
            return 'HIGH'
        elif 'medium acuity' in description_lower or 'medium' in description_lower:
            return 'MEDIUM'
        elif 'low acuity' in description_lower or 'low' in description_lower:
            return 'LOW'
        return 'MEDIUM'

    def _extract_emergency_status(self, description: str) -> str:
        description_lower = description.lower()
        if 'emergency status is yes' in description_lower or 'emergency: yes' in description_lower or 'emergency yes' in description_lower:
            return 'YES'
        elif 'emergency status is no' in description_lower or 'emergency: no' in description_lower or 'emergency no' in description_lower:
            return 'NO'
        return 'NO'

    def _extract_care_team_size(self, description: str) -> int:
        match = re.search(r'care team consists of (\d+) members', description.lower())
        if not match:
            match = re.search(r'care team of (\d+)', description.lower())
        if match:
            return int(match.group(1))
        return 3

    def _extract_special_considerations(self, description: str, safety_display: str):
        considerations = []
        desc_lower = description.lower() if description else ""
        if 'isolation' in desc_lower:
            considerations.append('Patient under isolation')
        if 'immunocompromised' in desc_lower:
            considerations.append('Patient is immunocompromised')
        if 'audit trail' in desc_lower or 'audit-trail' in desc_lower:
            considerations.append('Audit trail required')
        if 'remote monitoring' in desc_lower or 'remote-monitoring' in desc_lower:
            considerations.append('Remote monitoring required')
        if safety_display and 'shared' in safety_display.lower():
            considerations.append('Device is shared across beds')
        if 'special considerations include' in desc_lower:
            parts = re.split(r'(?i)special considerations include[:\s]*', description, maxsplit=1)
            if len(parts) > 1:
                special_section = parts[1].strip()
                special_section = re.split(r'[\.\n]', special_section, maxsplit=1)[0]
                for s in re.split(r'[;,]', special_section):
                    s = s.strip()
                    if s:
                        considerations.append(s[0].upper() + s[1:] if len(s) > 1 else s.upper())
        # dedupe preserving order
        seen = set()
        deduped = []
        for c in considerations:
            if c not in seen:
                deduped.append(c)
                seen.add(c)
        return deduped

    def _print_statistics(self, scenarios: List[PolicyScenario]):
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        device_types = {}
        for s in scenarios:
            device_types[s.device_type] = device_types.get(s.device_type, 0) + 1
        logger.info(f"Device Types ({len(device_types)} unique):")
        for dtype, count in sorted(device_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {dtype}: {count}")

        criticality = {}
        for s in scenarios:
            criticality[s.criticality] = criticality.get(s.criticality, 0) + 1
        logger.info("Criticality Levels:")
        for crit, count in sorted(criticality.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {crit}: {count} ({count/len(scenarios)*100:.1f}%)")

        emergency_count = sum(1 for s in scenarios if s.emergency_status == 'YES')
        logger.info(f"Emergency Scenarios: {emergency_count} ({emergency_count/len(scenarios)*100:.1f}%)")
        logger.info("="*60 + "\n")