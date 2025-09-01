import pandas as pd
from sqlalchemy import create_engine
from fastapi import FastAPI

app = FastAPI()

# Function to fetch table data from Postgres
@app.post("/fetch_table_from_postgres")
def fetch_table_from_postgres(table_name: str):
    """
    Connects to PostgreSQL and fetches the specified table into a pandas DataFrame.
    Args:
        table_name (str): The name of the table in Postgres.
    Returns:
        pd.DataFrame: Table data.
    """
    
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

    return df.to_dict(orient="records")