from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/resultados")
def obtener():
    df = pd.read_excel("resultado.xlsx")
    return df.to_dict(orient="records")
