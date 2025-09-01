# import pandas as pd
# from sqlalchemy import create_engine
# from fastapi import FastAPI

# app = FastAPI()

# # Function to fetch table data from Postgres
# @app.post("/fetch_table_from_postgres")
# def fetch_table_from_postgres(table_name: str):
#     """
#     Connects to PostgreSQL and fetches the specified table into a pandas DataFrame.
#     Args:
#         table_name (str): The name of the table in Postgres.
#     Returns:
#         pd.DataFrame: Table data.
#     """
    
#     schema="BHC"                     
#     user="postgres"                    
#     password="12345678"            
#     host="localhost"
#     port=5432
#     database="agentdata"   
    
#     # Create connection string
#     connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

#     # Create engine
#     engine = create_engine(connection_uri)

#     # SQL query
#     query = f'SELECT * FROM "{schema}".{table_name}'

#     # Read into DataFrame
#     df = pd.read_sql(query, engine)

#     return df.to_dict(orient="records")

# import pandas as pd
# from sqlalchemy import create_engine
# from fastapi import FastAPI
# import uvicorn
# from pyngrok import ngrok

# app = FastAPI()

# # Function to fetch table data from Postgres
# @app.post("/fetch_table_from_postgres")
# def fetch_table_from_postgres(table_name: str):
#     schema = "BHC"                     
#     user = "postgres"                    
#     password = "12345678"            
#     host = "localhost"
#     port = 5432
#     database = "agentdata"   
    
#     connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
#     engine = create_engine(connection_uri)
#     query = f'SELECT * FROM "{schema}".{table_name}'
#     df = pd.read_sql(query, engine)

#     return df.to_dict(orient="records")

# if __name__ == "__main__":
#     # ✅ Set your auth token here
#     ngrok.set_auth_token("325XiY4o8H2Pmx29j6ktr68iPnQ_573PDdVZDpAx1KHgTc4cq")
    
#     # Open an ngrok tunnel to port 8000
#     public_url = ngrok.connect(8000)
#     print("Public URL:", public_url)

#     uvicorn.run(app, host="0.0.0.0", port=8000)

import pandas as pd
from sqlalchemy import create_engine
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok

app = FastAPI()

# Function to fetch table data from Postgres
@app.post("/fetch_table_from_postgres")
def fetch_table_from_postgres(table_name: str):
    schema = "BHC"                     
    user = "postgres"                    
    password = "12345678"            
    host = "localhost"
    port = 5432
    database = "agentdata"   
    
    connection_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_uri)
    query = f'SELECT * FROM "{schema}".{table_name}'
    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")

if __name__ == "__main__":
    # 🔹 No need to set token explicitly since it’s already in ngrok.yml
    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8000)
