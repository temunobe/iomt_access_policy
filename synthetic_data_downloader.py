import csv
import random
import string

# Configuration
output_file = "/home/bsindala/projects/datasets/clinical_access_control_scenarios_1M.csv"
num_scenarios = 1_000_000

# Reference data
units = ["Oncology Unit", "Pediatric ICU", "Psychiatric Ward", "Cardiac Care Unit", "Outpatient Clinic"]
devices = [
    "CardiacMonitor", "InfusionPump", "Ventilator", "BloodPressureMonitor", "GlucoseMeter",
    "PulseOximeter", "EKGMachine", "CTScanner", "MRIScanner", "Defibrillator",
    "AnesthesiaMachine", "PatientMonitor", "ECMO", "DialysisMachine", "SmartBed",
    "TelemetrySystem", "IVPump"
]
statuses = ["active", "inactive", "maintenance"]
manufacturers = [
    "Snow, Johnson and Hernandez", "Cox Ltd", "Tran-Moore", "Rose, Andrews and Chen",
    "Harris-Roman", "Chavez, Marshall and Reynolds", "Vasquez, Gutierrez and Thomas",
    "Parker Industries", "Knight Medical", "Zhao Healthcare"
]
criticalities = ["ADMINISTRATIVE", "DIAGNOSTIC", "MONITORING", "LIFE_SUPPORTING", "CRITICAL"]
safety_codes = ["precaution", "alert", "notice"]
roles = ["Charge_Nurse", "Radiologist", "Fellow", "Perfusionist", "Intensivist", "Surgeon"]
acuity_levels = ["LOW", "MEDIUM", "HIGH"]
emergency_statuses = ["YES", "NO"]
special_considerations = [
    "Device is shared across beds", "Patient under isolation", "Patient is immunocompromised",
    "Audit trail required", "Remote monitoring required", "Device mobility required",
    "Requires calibration before use", "Device connected to EHR", "Device used for pediatric patient"
]

# CSV header
header = [
    "scenario_id", "description", "device_id", "device_type", "device_status",
    "manufacturer", "model_number", "device_name", "version", "location",
    "safety_code", "safety_display", "criticality", "access_policy"
]

# Helper: generate random version
def random_version():
    return f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"

# Helper: random model number
def random_model(prefix):
    return f"{prefix[:3].upper()}-{random.randint(100, 999)}"

# Generate XACML-style access policy
def generate_policy(scenario_id, device_id, role, emergency, acuity):
    return f'''<Policy PolicyId="AccessPolicy_{scenario_id}" RuleCombiningAlgId="urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:deny-overrides">
<Target>
<Subjects>
<SubjectMatch MatchId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
<AttributeValue>{role}</AttributeValue>
<SubjectAttributeDesignator AttributeId="role" DataType="http://www.w3.org/2001/XMLSchema#string" />
</SubjectMatch>
</Subjects>
<Resources>
<ResourceMatch MatchId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
<AttributeValue>{device_id}</AttributeValue>
<ResourceAttributeDesignator AttributeId="device-id" DataType="http://www.w3.org/2001/XMLSchema#string" />
</ResourceMatch>
</Resources>
<Actions>
<ActionMatch MatchId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
<AttributeValue>read</AttributeValue>
<ActionAttributeDesignator AttributeId="action-id" DataType="http://www.w3.org/2001/XMLSchema#string" />
</ActionMatch>
</Actions>
</Target>
<Rule RuleId="PermitAccess" Effect="Permit">
<Condition>
<Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:boolean-and">
<Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
<AttributeValue>{emergency}</AttributeValue>
<AttributeDesignator AttributeId="emergency-status" DataType="http://www.w3.org/2001/XMLSchema#string" />
</Apply>
<Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
<AttributeValue>{acuity}</AttributeValue>
<AttributeDesignator AttributeId="patient-acuity" DataType="http://www.w3.org/2001/XMLSchema#string" />
</Apply>
</Apply>
</Condition>
</Rule>
<Rule RuleId="DenyOthers" Effect="Deny" />
</Policy>'''

# Generation loop
with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for i in range(1, num_scenarios + 1):
        unit = random.choice(units)
        device_type = random.choice(devices)
        status = random.choice(statuses)
        manufacturer = random.choice(manufacturers)
        model = random_model(device_type)
        version = random_version()
        location = f"{unit} Room {random.randint(1, 20)}"
        safety_code = random.choice(safety_codes)
        safety_display = random.choice(special_considerations)
        criticality = random.choice(criticalities)
        role = random.choice(roles)
        emergency = random.choice(emergency_statuses)
        acuity = random.choice(acuity_levels)
        members = random.randint(3, 10)
        special_notes = ", ".join(random.sample(special_considerations, random.randint(1, 3)))

        prefix = unit.split()[0][:3].upper()
        device_prefix = device_type[:3].upper()
        scenario_id = f"{prefix}_{device_prefix}_{i:04d}"
        device_id = f"{device_type.lower()}-{unit.lower().replace(' ', '-')}-{i}"
        device_name = f"{unit} {device_type}"
        description = (
            f"A patient in the {unit} is being monitored using a {device_type}. "
            f"The patient has {acuity.lower()} acuity and emergency status is {emergency}. "
            f"The care team consists of {members} members. Special considerations include: {special_notes}."
        )
        policy = generate_policy(scenario_id, device_id, role, emergency, acuity)

        row = [
            scenario_id, description, device_id, device_type, status,
            manufacturer, model, device_name, version, location,
            safety_code, safety_display, criticality, policy
        ]
        writer.writerow(row)

print(f"✅ Successfully generated {num_scenarios:,} clinical access control scenarios to {output_file}")
