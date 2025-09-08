import pandas as pd
from sqlalchemy import create_engine
from fastapi import FastAPI, Request,APIRouter

app = FastAPI()
prefix_router = APIRouter(prefix="/book-appointments")



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
        "name": appointment_data["patient_info"]["name"],
        "age": appointment_data["patient_info"]["age"],
        "gender": appointment_data["patient_info"]["gender"],
        "appointment_date": appointment_data["date"],
        "doctor_name": appointment_data["doctor_details"]["Name"],
        "department": appointment_data["doctor_details"]["Department"],
        "time_slots": appointment_data["doctor_details"]["Available Slots"]
    }

    # Convert dict → DataFrame (single row)
    df = pd.DataFrame([row])

    # Append to Postgres table
    df.to_sql("Appointment_Table", engine, schema=schema, if_exists="append", index=False)

    print("✅ Appointment saved to Postgres!")
    return {"status": "success", "message": "Appointment saved"}
app.include_router(prefix_router)
# app = FastAPI()
# prefix_router = APIRouter(prefix="/book-appointments")

# @prefix_router.post("/save_appointment")
# async def save_appointment(request: Request):
#     payload = await request.json()
#     return save_appointment_to_postgres(payload)
