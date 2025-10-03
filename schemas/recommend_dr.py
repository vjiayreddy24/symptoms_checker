from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional
from datetime import date, datetime

# ---- For /recommend_doctors ----
class TestEvaluation(BaseModel):
    interpretation: str


class TestDetailedReport(BaseModel):
    detailed_report: TestEvaluation


class TestResult(BaseModel):
    testname: str
    evaluation_result: TestDetailedReport


class PatientInfo(BaseModel):
    id: int
    name: Optional[str] = None
    age: int
    gender: str
    email: EmailStr


class RecommendDoctorsRequest(BaseModel):
    patient_info: PatientInfo
    date: date
    test_result: List[TestResult]


# ---- For /save_appointment ----
class DoctorDetails(BaseModel):
    ID: str
    Name: Optional[str] = None
    Department: str
    Available_Slots: str


class SaveAppointmentRequest(BaseModel):
    patient_info: PatientInfo
    doctor_details: DoctorDetails
    date: date
    test_result: List[TestResult]


# ---- For update_appointment_load ----
class UpdateAppointmentLoadRequest(BaseModel):
    date: date
    doctor_details: DoctorDetails


# class AppointmentCreate(BaseModel):
#     appointment_id: str
#     patient_id: int
#     name: str
#     age: int
#     gender: str
#     test_taken: Dict[str, str] 
#     appointment_date: date
#     doctor_id: str
#     doctor_name: str
#     department: str
#     time_slots: str
#     booking_timestamp: datetime
#     to_email: EmailStr