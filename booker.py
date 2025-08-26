import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Appointment Scheduler")

# Standardization map
STANDARDIZATION_MAP = {
    "Minimal": [
        "Minimal", "Low Risk", "None", "Negative", "Unlikely", 
        "No significant symptoms", "Below threshold"
    ],
    "Mild": [
        "Mild", "Mild symptoms", "Slight", "Mild risk", "Positive (mild)"
    ],
    "Moderate": [
        "Moderate", "Moderately Severe", "Moderate risk", "Positive (moderate)", 
        "Somewhat likely", "Borderline", "At risk"
    ],
    "Severe": [
        "Severe", "High Risk", "Very Severe", "Strongly Positive", "Likely", 
        "Clinically significant", "Critical", "Probably severe",
        "High risk of eating disorder"
    ]
}

def standardize_interpretations(test_results: dict) -> dict:
    """
    Takes a dictionary of test results and returns standardized interpretations.
    Example:
    {"AQ-10":"Probably severe","EAT-28":"High risk of eating disorder"} 
    → {"AQ-10":"Severe","EAT-28":"Severe"}
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

# # Example test
# data = {"AQ-10": "Probably severe", "EAT-28": "High risk of eating disorder"}
# print(standardize_interpretations(data))

# ---------------- BOOK APPOINTMENT ---------------- #
def book_appointment(user_id: int,test_results: dict):
    DISORDER_MAPPING = {
        "AQ-10": "Autism Spectrum Disorder",
        "ASRS": "ADHD",
        "GAD-7": "Anxiety",
        "PHQ-9": "Depression",
        "Y-BOCS": "OCD",   
        "PCL-5": "PTSD",
        "AUDIT": "Alcohol Use Disorder",
        "DAST": "Drug Use Disorder",
        "Mood Disorder Questionnaire": "Bipolar Disorder",
        "MSI-BPD": "BPD",
        "EAT-26": "Eating Disorders"
    }

<<<<<<< HEAD
    # df = pd.read_csv(r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\UserData.csv")
    # df.columns = df.columns.str.strip()
=======
    df = pd.read_csv(r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\UserData.csv")
    df.columns = df.columns.str.strip()
>>>>>>> 609ae833649b5e40c5e85b03a6e0f9110ac7371d

    # if user_id not in df["User ID"].values:
    #     return {"error": "❌ User not found."}

    # user_record = df[df["User ID"] == user_id].iloc[0]

    # if pd.isna(user_record["Test Taken"]) or user_record["Test Taken"] == "":
    #     return {"error": f"📝 User {user_record['User Name']} has no test records. Please take a test first."}

    # test_results = json.loads(user_record["Test Taken"])

    test_results = standardize_interpretations(test_results)
    disorder_detected = [] 
    department = None

    for code, severity in test_results.items():
        if str(severity).strip().lower() == "severe" and code in DISORDER_MAPPING:
            disorder_detected.append(DISORDER_MAPPING[code])
            # department = "Psychiatry"
            # break

    if disorder_detected:
        department = "Psychiatry"
    else:
        disorder_detected = [DISORDER_MAPPING[code] for code in test_results if code in DISORDER_MAPPING]
        department = "Psychology"

    return {
        "user_id": user_id,
        "department": department,
        "disorder": disorder_detected
    }

# ---------------- FASTAPI ---------------- #
doctors_df = pd.read_csv(r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\Data\doctor_data.csv")
doctors_df.columns = doctors_df.columns.str.strip()

class AppointmentRequest(BaseModel):
    user_id: int
    date: str   # format YYYY-MM-DD
    test_results: dict = None  # Optional, not used in current logic

@app.post("/book_and_find_doctor/")
def book_and_find_doctor(req: AppointmentRequest):
    # Step 1: Get disorder + department from user’s record
    if req.test_results:  # ✅ If passed in request, standardize and use
        booking_info = book_appointment(req.user_id, req.test_results)
        if "error" in booking_info:
            return booking_info
        department = booking_info["department"]
        disorder = booking_info["disorder"]

        # # Handle case where disorder is a list (no severe case found)
        # if isinstance(disorder, list):
        #     disorder = disorder[0]  # pick the first one for filtering doctors

        # Step 2: Convert date → weekday
        try:
            date_obj = datetime.strptime(req.date, "%Y-%m-%d")
            weekday = date_obj.strftime("%a")  # Mon, Tue, etc.
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}

        # Step 3: Filter doctors
        filtered = doctors_df[
            (doctors_df["Specialization"].str.lower() == department.lower()) &
            (doctors_df["Disorder"].str.lower().isin([d.lower() for d in disorder])) &
            (doctors_df["Available Days"].str.contains(weekday))
        ]

        if filtered.empty:
            return doctors_df[
            (doctors_df["Specialization"].str.lower() == department.lower()) &
            (doctors_df["Available Days"].str.contains(weekday))]

        doctors = []
        for _, row in filtered.iterrows():
            doctors.append({
                "doctor_id": row["Doctor ID"],
                "doctor_name": row["Doctor Name"],
                "specialization": row["Specialization"],
                "disorder": row["Disorder"],
                "day": weekday,
                "available_slots": row["Available Slots"].split(",")
            })

        return {
            "user_id": req.user_id,
            "department": department,
            "disorder": disorder,
            "date": req.date,
            "day": weekday,
            "available_doctors": doctors
        }
    else:
        return {"error": "Test results are required for appointment. Under costruction for symptom based booking"}
