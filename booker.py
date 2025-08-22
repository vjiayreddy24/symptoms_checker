import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Appointment Scheduler")

# ---------------- BOOK APPOINTMENT ---------------- #
def book_appointment(user_id: int):
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

    df = pd.read_csv(r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\UserData.csv")
    df.columns = df.columns.str.strip()

    if user_id not in df["User ID"].values:
        return {"error": "❌ User not found."}

    user_record = df[df["User ID"] == user_id].iloc[0]

    if pd.isna(user_record["Test Taken"]) or user_record["Test Taken"] == "":
        return {"error": f"📝 User {user_record['User Name']} has no test records. Please take a test first."}

    test_results = json.loads(user_record["Test Taken"])
    disorder_detected, department = None, None

    for code, severity in test_results.items():
        if str(severity).strip().lower() == "severe" and code in DISORDER_MAPPING:
            disorder_detected = DISORDER_MAPPING[code]
            department = "Psychiatry"
            break

    if disorder_detected:
        disorder_name = disorder_detected
    else:
        disorder_name = [DISORDER_MAPPING[code] for code in test_results if code in DISORDER_MAPPING]
        department = "Psychology"

    return {
        "user_id": int(user_record["User ID"]),
        "department": department,
        "disorder": disorder_name
    }

# ---------------- FASTAPI ---------------- #
doctors_df = pd.read_csv(r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\Data\doctor_data.csv")
doctors_df.columns = doctors_df.columns.str.strip()

class AppointmentRequest(BaseModel):
    user_id: int
    date: str   # format YYYY-MM-DD

@app.post("/book_and_find_doctor/")
def book_and_find_doctor(req: AppointmentRequest):
    # Step 1: Get disorder + department from user’s record
    booking_info = book_appointment(req.user_id)
    if "error" in booking_info:
        return booking_info

    department = booking_info["department"]
    disorder = booking_info["disorder"]

    # Handle case where disorder is a list (no severe case found)
    if isinstance(disorder, list):
        disorder = disorder[0]  # pick the first one for filtering doctors

    # Step 2: Convert date → weekday
    try:
        date_obj = datetime.strptime(req.date, "%Y-%m-%d")
        weekday = date_obj.strftime("%a")  # Mon, Tue, etc.
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Step 3: Filter doctors
    filtered = doctors_df[
        (doctors_df["Specialization"].str.lower() == department.lower()) &
        (doctors_df["Disorder"].str.lower() == disorder.lower()) &
        (doctors_df["Available Days"].str.contains(weekday))
    ]

    if filtered.empty:
        return {
            "message": f"No doctors available in {department} for {disorder} on {weekday} ({req.date})."
        }

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
