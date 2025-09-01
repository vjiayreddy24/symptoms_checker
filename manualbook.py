import pandas as pd
from datetime import datetime

# Load doctor data
doctors_df = pd.read_csv(
    r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\CSV Data\doctors.csv"
)

class AppointmentRequest:
    def __init__(self, department: str, disorder: str, date: str):
        self.department = department
        self.disorder = disorder
        self.date = date

def find_doctors(req: AppointmentRequest):
    # Step 1: Convert date → weekday
    try:
        date_obj = datetime.strptime(req.date, "%Y-%m-%d")
        weekday = date_obj.strftime("%a")  # Mon, Tue, Wed...
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Step 2: Filter doctors by department + day
    filtered = doctors_df[
        (doctors_df["Department"].str.lower() == req.department.lower()) &
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
        "department": req.department,
        "disorder": req.disorder,
        "date": req.date,
        "day": weekday,
        "available_doctors": doctors
    }

# ------------------ TEST LOCALLY ------------------
if __name__ == "__main__":
    # Example test
    req = AppointmentRequest(
        department="Psychology",
        disorder="Heart Disease",
        date="2025-09-01"
    )

    result = find_doctors(req)
    print(result)
