import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request

# Load doctor data
doctors_df = pd.read_csv(
    r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\CSV Data\doctors.csv"
)

# Standardization map
STANDARDIZATION_MAP = {
    "Minimal": [
        "Minimal", "Low Risk", "None", "Negative", "Subclinical", "no problems","abstainer",
        "No significant symptoms", "Below threshold","Unlikely to need assessment","Unlikely BPD"
    ],
    "Mild": [
        "Mild", "Mild symptoms", "Slight","low level", "Mild risk", "Positive (mild)","Insufficient evidence for ADHD diagnosis",
        "low-risk","Unlikely PTSD","Negative screen","Low risk"
    ],
    "Moderate": [
        "Moderate", "Moderately Severe", "moderate level (likely abuse)", "Positive (moderate)", 
        "Somewhat likely", "Borderline range", "likely alcohol dependence","Possible risk",
        "At risk","May benefit from full diagnostic assessment","Positive screen for bipolar disorder"
    ],
    "Severe": [
        "Severe", "High Risk", "severe level", "Strongly Positive", "Likely", 
        "Clinically significant", "Critical", "Probably severe",
        "substantial level","Likely BPD",
        "High risk of eating disorder","Consistent with ADHD diagnosis",
        "hazardous or harmful use","Extreme",
        "Probable PTSD (clinical interview recommended)",

    ]
}

def standardize_interpretations(test_results: dict) -> dict:
    """
    Takes a dictionary of test results and returns standardized interpretations.
    Example:
    {"AQ-10":"Probably severe","EAT-28":"High risk of eating disorder"} → {"AQ-10":"Severe","EAT-28":"Severe"}
    """
    standardized = {}

    for test_code, interpretation in test_results.items():
        standardized_level = "Unknown"
        for standard, variants in STANDARDIZATION_MAP.items():
            if interpretation.strip().lower() in [v.lower() for v in variants]:
                standardized_level = standard
                break
        standardized[test_code] = standardized_level

    return standardized

# ------------------ CORE FUNCTION ------------------
def find_doctors(department: str, disorder: str, date: str):
    # Step 1: Convert date → weekday
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        weekday = date_obj.strftime("%a")  # Mon, Tue, Wed...
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Step 2: Filter doctors by department + day
    filtered = doctors_df[
        (doctors_df["Department"].str.lower() == department.lower()) &
        (doctors_df["Available Days"].str.contains(weekday))
    ]

    if filtered.empty:
        return {"message": "No doctors available for the given criteria."}

    # Step 3: Build response
    doctors = []
    for _, row in filtered.iterrows():
        doctors.append({
            "doctor_id": row["Doctor ID"],
            "doctor_name": row["Name"],
            "specialization": row["Department"],
            "day": weekday,
            "available_slots": row["Available Slots"].split(",")
        })

    return {
        "department": department,
        "disorder": disorder,
        "date": date,
        "day": weekday,
        "available_doctors": doctors
    }

# ------------------ FASTAPI APP ------------------
app = FastAPI()

@app.post("/manual_booking")
async def manual_booking(request: Request):
    payload = await request.json()
    
    # --- Extract patient info ---
    patient_info = payload.get("patient_info", {})
    name = patient_info.get("name")
    age = patient_info.get("age")
    gender = patient_info.get("gender")
    
    # --- Extract appointment date ---
    date = payload.get("date")
  
    # --- Extract department ---
    department=payload.get("department")

    # --- Extract department ---
    disorder=payload.get("disorder")

    # --- Call doctor finder ---
    doctor_recommendations = find_doctors(department, disorder, date)

    return {
        "patient_name": name,
        "age": age,
        "gender": gender,
        "department": department,
        "date": date,
        "recommendations": doctor_recommendations
    }

# ------------------ TEST LOCALLY ------------------
if __name__ == "__main__":
    # Example payload
    payload = {
        "patient_info": {"name": "asd", "age": 22, "gender": "female"},
        "symptoms": "",
        "test_result": {
            "testname": "ASRS",
            "evaluation_result": {
                "patient_summary": "Your responses indicate possible ADHD symptoms.",
                "detailed_report": {
                    "part_a_score": 13,
                    "part_b_score": 20,
                    "score": 33,
                    "interpretation": "Severe ADHD indicators",
                    "risk_flags": [],
                    "clinical_notes": ["Part A ≥14 suggests ADHD diagnostic consistency"]
                }
            }
        },
        "date": "2025-09-01",
        "department": "Psychology",
        "disorder": "ADHD"
    }

    # Simulate FastAPI behavior locally
    dept = "Psychiatry" if "severe" in payload["test_result"]["evaluation_result"]["detailed_report"]["interpretation"].lower() else "Psychology"
    result = find_doctors(dept, payload["test_result"]["testname"], payload["date"])
    print(result)
