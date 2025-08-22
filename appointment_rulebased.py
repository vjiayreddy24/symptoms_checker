from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

app = FastAPI(title="Appointment Scheduler")

# Load doctor data
doctors_df = pd.read_csv(r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\Data\doctor_data.csv")  # Save your table as CSV (Doctor ID,...)

class AppointmentRequest(BaseModel):
    department: str
    disorder: str
    date: str  # format: YYYY-MM-DD

@app.post("/find_doctors/")
def find_doctors(req: AppointmentRequest):
    # Step 1: Convert date → weekday (e.g. Monday, Tue...)
    try:
        date_obj = datetime.strptime(req.date, "%Y-%m-%d")
        weekday = date_obj.strftime("%a")  # Mon, Tue, Wed, etc.
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Step 2: Filter doctors by department + disorder + day
    filtered = doctors_df[
        (doctors_df["Specialization"].str.lower() == req.department.lower()) &
        (doctors_df["Disorder"].str.lower() == req.disorder.lower()) &
        (doctors_df["Available Days"].str.contains(weekday))
    ]

    if filtered.empty:
        filtered = doctors_df[
            (doctors_df["Specialization"].str.lower() == req.department.lower()) &
            (doctors_df["Available Days"].str.contains(weekday))]
        return filtered[:3]

    
    # Step 3: Build response
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
        "department": req.department,
        "disorder": req.disorder,
        "date": req.date,
        "day": weekday,
        "available_doctors": doctors
    }  