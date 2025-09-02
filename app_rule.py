import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import nest_asyncio
import uvicorn
from datetime import datetime

# Patch asyncio for Jupyter
nest_asyncio.apply()

# -----------------------------
# Load Data
# -----------------------------
doctors_df = pd.read_csv(
    r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\Data\doctors.csv"
)
appointments_data = pd.read_csv(
    r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\CSV_Data\doctor_appointment_load.csv"
)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# 1️⃣ Endpoint → Get Available Doctors
@app.get("/available_doctors/")
def get_available_doctors(date: str, department: str, disorder: str = None):
    try:
        weekday = datetime.strptime(date, "%Y-%m-%d").strftime("%a")

        dept_docs = doctors_df[doctors_df["Department"].str.lower() == department.lower()]
        if dept_docs.empty:
            return {"message": f"No doctors found in {department}"}

        dept_docs = dept_docs[dept_docs["Available Days"].apply(lambda d: weekday in d)]
        if dept_docs.empty:
            return {"message": f"No {department} doctors available on {date} ({weekday})"}

        if date in appointments_data.columns:
            merged = dept_docs.merge(
                appointments_data[["Doctor ID", date]],
                on="Doctor ID",
                how="left"
            )
            merged = merged.rename(columns={date: "Appointments"})
        else:
            merged = dept_docs.copy()
            merged["Appointments"] = "NA"

        merged["Appointments"] = merged["Appointments"].replace("Leave", "On Leave")

        return {
            "date": date,
            "department": department,
            "disorder": disorder,
            "available_doctors": merged.to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 2️⃣ Helper Function → Get Availability with Leave Handling
def get_doctor_availability(user_date, common_doctors: set):
    appointments_df = pd.read_csv(
        r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\CSV Data\doctor_appointment_load.csv"
    )
    leaves_df = pd.read_csv(
        r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\CSV Data\Leavedata.csv"
    )
    leaves_df["Leave_Dates"] = pd.to_datetime(leaves_df["Leave_Dates"], format="%d-%m-%Y")

    user_date = pd.to_datetime(user_date, format="%Y-%m-%d")

    doctors_on_leave = leaves_df.loc[
        leaves_df["Leave_Dates"] == user_date, "Doctor_ID"
    ].tolist()

    date_str = user_date.strftime("%Y-%m-%d (%A)")
    if date_str not in appointments_df.columns:
        raise ValueError(f"Date {date_str} not found in appointment load table")

    result_df = appointments_df[["Doctor ID", "Doctor Name", "Department", date_str]].copy()
    result_df.rename(columns={date_str: "Load"}, inplace=True)

    result_df = result_df[result_df["Doctor ID"].isin(common_doctors)]
    result_df = result_df[~result_df["Doctor ID"].isin(doctors_on_leave)]
    result_df["Load"] = pd.to_numeric(result_df["Load"], errors="coerce").fillna(0)
    result_df = result_df[result_df["Load"] > 0]

    return dict(zip(result_df["Doctor ID"], result_df["Load"]))


# 3️⃣ Helper Function → Compute Doctor Scores
def compute_doctor_scores(doctor_availability: dict, patient_gender: str):
    max_daily_capacity = 60
    doctors_master = pd.read_csv(
        r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\Data\doctors.csv"
    )
    doctor_gender_map = doctors_master.set_index("Doctor ID")["Gender"].to_dict()

    rows = []
    for doctor_id, load_today in doctor_availability.items():
        A = 1 if load_today < max_daily_capacity else 0
        L = max(0, 1 - (load_today / max_daily_capacity))
        doctor_gender = doctor_gender_map.get(doctor_id, None)
        G = 1 if doctor_gender == patient_gender else 0.5 if doctor_gender else 0.5
        FinalScore = (0.4 * A) + (0.4 * L) + (0.2 * G)

        rows.append({
            "Doctor ID": doctor_id,
            "AppointmentsToday": load_today,
            "Availability": A,
            "Load Balancing": L,
            "Gender Affinity": G,
            "FinalScore": FinalScore
        })

    top_dr = pd.DataFrame(rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
    top_dr_full = top_dr.merge(doctors_master, on="Doctor ID", how="left")

    return top_dr_full[["Doctor ID", "Name", "Gender", "Department",
                        "Available Days", "Available Slots", "FinalScore"]]


# 4️⃣ New Endpoint → Ranked Doctors
@app.get("/ranked_doctors/")
def get_ranked_doctors(date: str, department: str, patient_gender: str):
    try:
        # Step 1 → Get available doctors list
        available_response = get_available_doctors(date=date, department=department)
        if "available_doctors" not in available_response:
            return available_response  # message error passthrough

        available_doctors = available_response["available_doctors"]
        doctor_ids = {doc["Doctor ID"] for doc in available_doctors}

        # Step 2 → Get availability (exclude leave, >0 load)
        availability_dict = get_doctor_availability(date, doctor_ids)
        if not availability_dict:
            return {"message": f"No {departme nt} doctors with load > 0 on {date}"}

        # Step 3 → Compute scores
        ranked_df = compute_doctor_scores(availability_dict, patient_gender)

        return ranked_df.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
