import pandas as pd
from sqlalchemy import create_engine

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
        "Probable PTSD (clinical interview recommended)","Severe ADHD indicators"

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

import ast
from fastapi import FastAPI, Request,APIRouter

app = FastAPI()
prefix_router = APIRouter(prefix="/book-appointment")

@prefix_router.post("/recommend_doctors")
async def recommend_doctors(request: Request):

    payload = await request.json()

    # --- Extract patient info ---
    patient_info = payload.get("patient_info", {})
    user_age = patient_info.get("age")
    patient_gender = patient_info.get("gender")

    # --- Extract appointment date ---
    user_date = payload.get("date")

    # --- Extract test result in desired format ---
    test_result = payload.get("test_result", {})
    testname = test_result.get("testname")
    interpretation = (
        test_result.get("evaluation_result", {})
        .get("detailed_report", {})
        .get("interpretation")
    )

    # Build user_test dict → {"ASRS": "Severe ADHD indicators"}
    user_test = {testname: interpretation} if testname and interpretation else {}
    user_test = standardize_interpretations(user_test)

    # Loop over multiple tables and store with table names
    table_names = ["test_segment", "appointments_df", "age_segment","dept_segment","leaves_df","doctors_df"]

    dfs = {name: fetch_table_from_postgres(name) for name in table_names}

    # Access like:
    appointments_df=dfs["appointments_df"]
    leaves_df=dfs["leaves_df"]
    age_segment=dfs["age_segment"]
    dept_segment=dfs["dept_segment"]
    doctors_df=dfs["doctors_df"]
    test_segment=dfs["test_segment"]

    # -----------------------
    # Department Mapping (NEW)
    # -----------------------
    if any(result.lower() == "severe" for result in user_test.values()):
        user_department = "Psychiatry"
    else:
        user_department = "Psychology"

    # -----------------------
    # 1. Age → doctor list
    # -----------------------
    def get_age_segment(age: int) -> str:
        if 2 <= age <= 4:
            return "2-4"
        elif 5 <= age <= 12:
            return "5-12"
        elif 13 <= age <= 18:
            return "13-18"
        elif 19 <= age <= 30:
            return "19-30"
        elif 31 <= age <= 45:
            return "31-45"
        elif 46 <= age <= 60:
            return "46-60"
        else:
            return "61+"

    user_age_segment = get_age_segment(user_age)
    age_doctors = age_segment.loc[age_segment["age_group"] == user_age_segment, "doctor_id"].values[0]
    if isinstance(age_doctors, str):
        age_doctors = ast.literal_eval(age_doctors)
    age_doctors = set(age_doctors)

    # -----------------------
    # 2. Test → doctor list
    # -----------------------
    test_segment["test_combination"] = test_segment["test_combination"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    test_segment["doctors"] = test_segment["doctors"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def get_doctors_for_test(user_test_dict, test_segment):
        matching_doctors = []
        for user_test in user_test_dict.keys():
            for _, row in test_segment.iterrows():
                if user_test in row["test_combination"]:
                    matching_doctors.extend(row["doctors"])
        return list(set(matching_doctors))  # remove duplicates

    test_doctors = set(get_doctors_for_test(user_test, test_segment))

    # -----------------------
    # 3. Department → doctor list
    # -----------------------
    dept_segment["doctors"] = dept_segment["doctors"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    dept_doctors = dept_segment.loc[
        dept_segment["department"].str.lower() == user_department.lower(), "doctors"
    ].values[0]
    dept_doctors = set(dept_doctors)

    # -----------------------
    # Final: Intersection
    # -----------------------
    common_doctors = age_doctors & test_doctors & dept_doctors

    availability = get_doctor_availability(appointments_df,leaves_df,user_date, common_doctors)

    scores_df = compute_doctor_scores(doctors_df,availability, patient_gender)
    return scores_df

app.include_router(prefix_router)

def fetch_table_from_postgres(table_name: str):
    schema="BHC"
    user="postgres"
    password="12345678"
    host="localhost"
    port=5432
    database="agentdata"

    # Create connection string
    connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # Create engine
    engine = create_engine(connection_uri)

    # SQL query
    query = f'SELECT * FROM "{schema}".{table_name}'

    # Read into DataFrame
    df = pd.read_sql(query, engine)

    return df

def get_doctor_availability(appointments_df,leaves_df,user_date, common_doctors: set):

    leaves_df["Leave_Dates"] = pd.to_datetime(leaves_df["Leave_Dates"], format="%d-%m-%Y")

    # Convert input date
    user_date = pd.to_datetime(user_date, format="%Y-%m-%d")

    # Step 1: find doctors on leave that day
    doctors_on_leave = leaves_df.loc[
        leaves_df["Leave_Dates"] == user_date, "Doctor_ID"
    ].tolist()

    # Step 2: get column name that matches user_date in appointment load
    date_str = user_date.strftime("%Y-%m-%d (%A)")
    if date_str not in appointments_df.columns:
        raise ValueError(f"Date {date_str} not found in appointment load table")

    # Step 3: filter appointment load for that date
    result_df = appointments_df[
        ["Doctor ID", "Doctor Name", "Department", date_str]
    ].copy()
    result_df.rename(columns={date_str: "Load"}, inplace=True)

    # Step 4: filter only common_doctors
    result_df = result_df[result_df["Doctor ID"].isin(common_doctors)]

    # Step 5: remove doctors on leave
    result_df = result_df[~result_df["Doctor ID"].isin(doctors_on_leave)]

    # Step 6: convert "Leave" text to 0 load
    result_df["Load"] = pd.to_numeric(result_df["Load"], errors="coerce").fillna(0)

    # Step 7: keep only doctors with load > 0
    result_df = result_df[result_df["Load"] > 0]

    # Step 8: return dictionary
    return dict(zip(result_df["Doctor ID"], result_df["Load"]))


def compute_doctor_scores(doctors_df,doctor_availability: dict, patient_gender: str):

    max_daily_capacity = 60
    doctor_gender_map = doctors_df.set_index("Doctor ID")["Gender"].to_dict()

    rows = []
    for doctor_id, load_today in doctor_availability.items():

        # Availability Score
        A = 1 if load_today < max_daily_capacity else 0

        # Load Balancing Score
        L = max(0, 1 - (load_today / max_daily_capacity))

        # Gender Affinity Score
        doctor_gender = doctor_gender_map.get(doctor_id, None)
        if doctor_gender is None:
            G = 0.5  # default if gender missing
        else:
            G = 1 if doctor_gender == patient_gender else 0.5

        # Final Score (weighted sum, you can adjust weights)
        FinalScore = (0.4 * A) + (0.4 * L) + (0.2 * G)

        rows.append({
            "Doctor ID": doctor_id,
            "AppointmentsToday": load_today,
            "Max load": A,
            "Load Balancing": L,
            "Gender affinity": G,
            "FinalScore": FinalScore
        })
    top_dr=pd.DataFrame(rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)

    # Merge with doctor details
    top_dr_full = top_dr.merge(doctors_df, on="Doctor ID", how="left")

    return top_dr_full[["Name", "Gender", "Department","Available Days", "Available Slots", "FinalScore"]]

import pandas as pd
from sqlalchemy import create_engine
from fastapi import Request
import uuid
from datetime import datetime

@prefix_router.post("/save_appointment")
async def save_appointment_to_postgres(request: Request):

    appointment_data = await request.json()

    schema = "BHC"                     
    user = "postgres"                    
    password = "12345678"            
    host = "localhost"
    port = 5432
    database = "agentdata"   
    
    # Create connection string
    connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_uri)

    # --- Extract data from JSON payload ---
    # appointment_data = request.json()  # payload dict

    row = {
        "appointment_id": str(uuid.uuid4()),
        "patient_id": appointment_data["patient_info"]["id"], 
        "name": appointment_data["patient_info"]["name"],
        "age": appointment_data["patient_info"]["age"],
        "gender": appointment_data["patient_info"]["gender"],
        "appointment_date": appointment_data["date"],
        "doctor_name": appointment_data["doctor_details"]["Name"],
        "department": appointment_data["doctor_details"]["Department"],
        "time_slots": appointment_data["doctor_details"]["Available Slots"],
        "booking_timestamp": datetime.now() 
    }

    # Convert dict → DataFrame (single row)
    df = pd.DataFrame([row])

    # Append to Postgres table
    df.to_sql("appointment_table", engine, schema=schema, if_exists="append", index=False)
    return {"status": "success", "message": "Appointment saved"}


def format_date_with_day(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%Y-%m-%d (%A)")

from fastapi import APIRouter, Request, HTTPException
from sqlalchemy import create_engine, text
from datetime import datetime

@prefix_router.post("/update_appointment_load")
async def update_appointment_load(request: Request):

    update_data = await request.json()
    date=update_data["date"]
    doctor_id=update_data["doctor_details"]["ID"]
    formatted_date = format_date_with_day(date)

        # Convert the date to 'yyyy-mm-dd' format
    try:
        formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")                    
    user = "postgres"                    
    password = "12345678"            
    host = "localhost"
    port = 5432
    database = "agentdata"   
    
    # Create connection string
    connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_uri)

        # Build and execute the SQL query
    query = text("""
        UPDATE "BHC".appointment_load
        SET appointment_load = (
            CASE 
                WHEN appointment_load NOT IN ('Not Applicable', 'Leave') 
                THEN (COALESCE(appointment_load::integer, 0) + 1)::text
                ELSE appointment_load
            END
        )
        WHERE doctor_id = :doctor_id AND date = :date;
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"doctor_id": doctor_id, "date": formatted_date})
        conn.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Doctor ID not found")

    return {"message": f"✅ Updated Doctor {doctor_id} on {formatted_date}"}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    
app.include_router(prefix_router)