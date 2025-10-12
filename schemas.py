# schemas.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PolicyScenario:
    """Represents a single training example"""
    scenario_id: str
    description: str
    device_id: str
    device_type: str
    device_status: str
    manufacturer: str
    model_number: str
    device_name: str
    version: str
    location: str
    safety_code: str
    safety_display: str
    criticality: str
    access_policy: str
    patient_acuity: Optional[str] = None
    emergency_status: Optional[str] = None
    care_team_size: Optional[int] = None
    special_considerations: Optional[List[str]] = None