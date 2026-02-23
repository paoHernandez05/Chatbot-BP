from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Carga las variables desde .env
load_dotenv()

# Obtiene la clave
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Información base de presión arterial
with open("documentsBP.txt", "r", encoding="utf-8") as f:
    hipertension_info = f.read()

# Información base de glucosa
with open("documentsGlucose.txt", "r", encoding="utf-8") as f:
    glucosa_info = f.read()


# División en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

#Chunks para presión arterial
chunks_presion = splitter.split_documents([Document(page_content=hipertension_info)])

#Chunks para glucosa
chunks_glucosa = splitter.split_documents([Document(page_content=glucosa_info)])


# Modelo
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Template para presión
template_presion = """Eres un asistente médico especializado en hipertensión. Responde de forma clara, confiable y breve.
Si la pregunta no es relevante, indícalo con amabilidad.
"""

# Template para glucosa
template_glucosa = """Eres un asistente médico especializado en glucosa en sangre y diabetes. Responde de forma clara, confiable y breve.  
Si la pregunta no es relevante, indícalo con amabilidad."""

# Prompt
prompt = ChatPromptTemplate.from_template("""{template}

Contexto:
{context}

Pregunta:
{input}
""")


# Cadena RAG
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

def responder_pregunta_presion(pregunta: str) -> str:
    respuesta = chain.invoke({
        "template": template_presion,
        "input": pregunta,
        "context": chunks_presion
    })
    return respuesta


def responder_pregunta_glucosa(pregunta: str) -> str:
    respuesta = chain.invoke({
        "template": template_glucosa,
        "input": pregunta,
        "context": chunks_glucosa
    })
    return respuesta
