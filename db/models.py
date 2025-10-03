import uuid
from sqlalchemy import Column, Integer, String, Date, Text, Float, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from .connection import Base, engine, SessionLocal
from sqlalchemy import PrimaryKeyConstraint

# class User(Base):
#     __tablename__ = "users"
#     __table_args__ = {"schema": "test"} 
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, nullable=False)
#     password = Column(String, nullable=True)  # you’ll need to hash manually

# class PatientProfile(Base):
#     __tablename__ = "patient_profiles"
#     __table_args__ = {"schema": "test"} 
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"))
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     age = Column(Integer)
#     gender = Column(String(20))
#     date_of_birth = Column(Date)
#     address = Column(Text)
#     city = Column(String(100))
#     state = Column(String(100))
#     zipcode = Column(String(20))
#     phone = Column(String(20))
#     email = Column(String(100))
#     emergency_first_name = Column(String(100))
#     emergency_last_name = Column(String(100))
#     emergency_phone = Column(String(20))
#     relationship = Column(String(50))
#     personal_history = Column(Text)
#     family_history = Column(Text)

class Appointment(Base):
    __tablename__ = "appointment_table"
    __table_args__ = {"schema": "test"} 
    appointment_id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    test_taken = Column(JSON)  # JSON type instead of dumping string
    appointment_date = Column(Date, nullable=False)
    doctor_id = Column(String, nullable=False)
    doctor_name = Column(String, nullable=False)
    department = Column(String, nullable=False)
    time_slots = Column(String, nullable=False)
    booking_timestamp = Column(DateTime, default=datetime.utcnow)

class AppointmentLoad(Base):
    __tablename__ = "appointment_load"
    __table_args__ = (
        PrimaryKeyConstraint("doctor_id", "date", name="pk_appointment_load"),
        {"schema": "test"}
    )
    doctor_id = Column(String(16), ForeignKey("test.doctors_df.doctor_id"))
    doctor_name = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    appointment_load = Column(String, default="0")

class AgeSegment(Base):
    __tablename__ = "age_segment"
    __table_args__ = {"schema": "test"} 
    age_group = Column(String(16), primary_key=True, index=True)
    doctor_id = Column(Text, nullable=False)  # Comma-separated doctor IDs

class DepartmentSegment(Base):
    __tablename__ = "dept_segment"
    __table_args__ = {"schema": "test"} 
    department = Column(String(30), primary_key=True, index=True)
    doctor_id = Column(Text, nullable=False)  # Comma-separated doctor IDs

class TestSegment(Base):
    __tablename__ = "test_segment"
    __table_args__ = {"schema": "test"} 
    test_combination = Column(Text, primary_key=True, index=True)
    doctor_id = Column(Text, nullable=False)  # Comma-separated doctor IDs

class DoctorDF(Base):
    __tablename__ = "doctors_df"
    __table_args__ = {"schema": "test"} 
    doctor_id = Column(String(16), primary_key=True, index=True)  
    name = Column( String, nullable=False)
    gender = Column( String, nullable=False)
    department = Column( String, nullable=False)
    available_days = Column( String, nullable=False)
    available_slots = Column(String, nullable=False)