import pandas as pd
from sqlalchemy import create_engine,text
from datetime import datetime, timedelta
import ast
from fastapi import FastAPI,APIRouter, Request, HTTPException
import uuid
import json
# from db.models import Appointment,AppointmentLoad,AgeSegment,DepartmentSegment,DoctorDF,TestSegment
from schemas.recommend_dr import RecommendDoctorsRequest,SaveAppointmentRequest,UpdateAppointmentLoadRequest,DoctorDetails,PatientInfo
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

schema = "BHC"                       
user = "mhealthadmin"                    
password = "mhealth*2025"            
host = "57.159.27.80"
port = 5432
database = "mhealth_db" 

router = APIRouter(prefix="/book-appointment", tags=["book-appointment"])

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

app = FastAPI()
prefix_router = APIRouter(prefix="/book-appointment")

@prefix_router.post("/recommend_doctors")
async def recommend_doctors(payload: RecommendDoctorsRequest):

    # payload = await request.json()

    # # --- Extract patient info ---
    # patient_info = payload.get("patient_info", {})
    # user_age = patient_info.get("age")
    # patient_gender = patient_info.get("gender")

    # # --- Extract appointment date ---
    # user_date = payload.get("date")

    # # --- Extract test result in desired format ---
    # test_result = payload.get("test_result", {})
    # testname = test_result.get("testname")
    # interpretation = (
    #     test_result.get("evaluation_result", {})
    #     .get("detailed_report", {})
    #     .get("interpretation")
    # )
        # --- Extract test results ---
    # test_results = payload.get("test_result", [])

    patient_info = payload.patient_info
    user_age = patient_info.age
    patient_gender = patient_info.gender
    user_date = payload.date
    test_results = payload.test_result

    user_test = {}

    for test in test_results:
        testname = test.get("testname")
        interpretation = (
            test.get("evaluation_result", {})
            .get("detailed_report", {})
            .get("interpretation")
        )
        if testname and interpretation:
            user_test[testname] = interpretation

    # # Build user_test dict → {"ASRS": "Severe ADHD indicators"}
    # user_test = {testname: interpretation} if testname and interpretation else {}
    user_test = standardize_interpretations(user_test)

    # Loop over multiple tables and store with table names
    table_names = ["test_segment", "appointment_load", "age_segment","dept_segment","doctors_df"]

    dfs = {name: fetch_table_from_postgres(name) for name in table_names}

    # Access like:
    appointments_df=dfs["appointment_load"]
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

    availability = get_doctor_availability(appointments_df,user_date, common_doctors)

    scores_df = compute_doctor_scores(doctors_df,availability, patient_gender,user_date)
    return scores_df

app.include_router(prefix_router)

def fetch_table_from_postgres(table_name: str): 

    # Create connection string
    connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # Create engine
    engine = create_engine(connection_uri)

    # SQL query
    query = f'SELECT * FROM "{schema}".{table_name}'

    # Read into DataFrame
    df = pd.read_sql(query, engine)

    return df

def get_doctor_availability(appointments_df, user_date, common_doctors: set):
    # Convert input date to datetime
    user_date = pd.to_datetime(user_date, format="%Y-%m-%d")

    # Filter appointments for the given date
    filtered_df = appointments_df[pd.to_datetime(appointments_df["date"]) == user_date].copy()

    # Keep only doctors in the common_doctors set
    filtered_df = filtered_df[filtered_df["doctor_id"].isin(common_doctors)]

    # Remove doctors marked as 'Leave' or 'Not Applicable'
    filtered_df = filtered_df[
        ~filtered_df["appointment_load"].isin(["Leave", "Not Applicable"])
    ]

    # Convert 'appointment_load' to numeric, coercing errors to NaN, then fill with 0
    filtered_df["Load"] = pd.to_numeric(filtered_df["appointment_load"], errors="coerce").fillna(0)

    # Keep only doctors with load > 0
    available_doctors = filtered_df[filtered_df["Load"] > 0]

    # Create a dictionary {doctor_id: load}
    result = dict(zip(available_doctors["doctor_id"], available_doctors["Load"]))

    return result

def expand_timeslot(slot_str, interval_minutes=20):
    """Expand a timeslot string like '12:00-13:00' into intervals."""
    start_str, end_str = slot_str.split("-")
    start = datetime.strptime(start_str.strip(), "%H:%M")
    end = datetime.strptime(end_str.strip(), "%H:%M")

    slots = []
    while start < end:
        next_time = start + timedelta(minutes=interval_minutes)
        if next_time > end:
            break
        slots.append(f"{start.strftime('%H:%M')}-{next_time.strftime('%H:%M')}")
        start = next_time
    return slots


def compute_doctor_scores(doctors_df, doctor_availability: dict, patient_gender: str, user_date: str):
    """
    doctors_df: DataFrame of doctor details
    doctor_availability: dict {doctor_id: appointments_today}
    patient_gender: 'Male'/'Female'
    user_date: date string in 'YYYY-MM-DD'
    """

    max_daily_capacity = 60
    doctor_gender_map = doctors_df.set_index("doctor_id")["gender"].to_dict()

    # Get 3-letter day code for given date
    day_code = datetime.strptime(user_date, "%Y-%m-%d").strftime("%a")  # e.g. 'Mon', 'Tue'

    rows = []
    for doctor_id, load_today in doctor_availability.items():
        # Availability Score
        A = 1 if load_today < max_daily_capacity else 0

        # Load Balancing Score
        L = max(0, 1 - (load_today / max_daily_capacity))

        # Gender Affinity Score
        doctor_gender = doctor_gender_map.get(doctor_id, None)
        if doctor_gender is None:
            G = 0.5
        else:
            G = 1 if doctor_gender == patient_gender else 0.5

        # Final Score
        FinalScore = (0.4 * A) + (0.4 * L) + (0.2 * G)

        rows.append({
            "doctor_id": doctor_id,
            "AppointmentsToday": load_today,
            "Max load": A,
            "Load Balancing": L,
            "Gender affinity": G,
            "FinalScore": FinalScore
        })

    top_dr = pd.DataFrame(rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)

    # Merge with doctor details
    top_dr_full = top_dr.merge(doctors_df, on="doctor_id", how="left")

    # Extract slots only for the given day
    expanded_slots = []
    for _, row in top_dr_full.iterrows():
        days = row["available_days"].split(",")
        slots = row["available_slots"].split(",")

        intervals_today = []
        if day_code in days:
            indices = [i for i, d in enumerate(days) if d == day_code]
            for idx in indices:
                time_range = slots[idx]
                intervals_today.extend(expand_timeslot(time_range))

        expanded_slots.append(intervals_today)

    top_dr_full["Timeslots on " + user_date] = expanded_slots

    return top_dr_full[[
        "doctor_id", "name", "gender", "department","available_days", "available_slots",
         "Timeslots on " + user_date, "FinalScore"
    ]]
EMAIL_HOST_USER = 'sswain@quantum-i.ai'
EMAIL_HOST_PASSWORD = 'jcvklhrggkdwljge'

@prefix_router.post("/save_appointment")
async def save_appointment_to_postgres(payload: SaveAppointmentRequest):

    # appointment_data = await request.json()

    # Create connection string
    connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_uri)

    # Serialize test_taken to JSON string
    # test_taken_json = json.dumps(test_taken)
    appointment_id=str(uuid.uuid4())
    pname=payload.patient_info.name
    doctor_name=payload.doctor_details.Name
    department=payload.doctor_details.Department
    time_slots=payload.doctor_details.Available_Slots
    appointment_date=payload.date,
    to_email=payload.patient_info.email

    row = {
        "appointment_id": appointment_id,
        "patient_id": payload.patient_info.id,
        "name": payload.patient_info.name,
        "age": payload.patient_info.age,
        "gender": payload.patient_info.gender,
        "test_taken": payload.test_result,  # stays dict → JSON automatically
        "appointment_date": payload.date,
        "doctor_id": payload.doctor_details.ID,
        "doctor_name": payload.doctor_details.Name,
        "department": payload.doctor_details.Department,
        "time_slots": payload.doctor_details.Available_Slots,
        "booking_timestamp": datetime.now() 
    }

    # Convert dict → DataFrame (single row)
    df = pd.DataFrame([row])

    # Append to Postgres table
    df.to_sql("appointment_table", engine, schema=schema, if_exists="append", index=False)
    print("Appointment saved")

    df=update_appointment_load(UpdateAppointmentLoadRequest(
        date=payload.date,
        doctor_details=payload.doctor_details
    ))
    print("updation",df)

    # --- Email content ---
    subject = "Appointment Confirmation - Synerza Healthcare"
    body = f"""
Dear {pname},

Your appointment has been successfully confirmed.

📄 Appointment Details:
- Appointment ID: {appointment_id}
- Doctor: Dr. {doctor_name}
- Department: {department}
- Date: {appointment_date}
- Time: {time_slots}

If you have any questions, please contact us at 040-0000-0000.

Best regards,  
Synerza Healthcare 
appointments@ankurahospitals.com
""".strip()
    
    # --- Build the email ---
    msg = MIMEMultipart()
    msg['From'] = EMAIL_HOST_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # --- Send ---
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
            server.send_message(msg)
        return {"status": "success", "message": f"Mail sent to {to_email}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to send email: {str(e)}"}


def format_date_with_day(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%Y-%m-%d (%A)")

# @prefix_router.post("/update_appointment_load")
async def update_appointment_load(update_data: UpdateAppointmentLoadRequest):

    date = update_data.date
    doctor_id = update_data.doctor_details.ID
    formatted_date = format_date_with_day(date)

        # Convert the date to 'yyyy-mm-dd' format
    try:
        formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")                    
    
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
            raise HTTPException(status_code=404, detail="doctor_id not found")

    return {"message": f"✅ Updated Doctor {doctor_id} on {formatted_date}"}
    
app.include_router(prefix_router)