from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core import responder_pregunta_presion, responder_pregunta_glucosa

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes limitar esto a ["http://127.0.0.1:5500"] si gustas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PreguntaRequest(BaseModel):
    pregunta: str

@app.post("/preguntarBP")
def preguntar(data: PreguntaRequest):
    try:
        respuesta = responder_pregunta_presion(data.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preguntarGlucosa")
def preguntar(data: PreguntaRequest):
    try:
        respuesta = responder_pregunta_glucosa(data.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
