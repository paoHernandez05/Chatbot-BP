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

# Información base
with open("documents.txt", "r", encoding="utf-8") as f:
    hipertension_info = f.read()

# División en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents([Document(page_content=hipertension_info)])

# Modelo
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Eres un asistente médico especializado en hipertensión. Responde de forma clara, confiable y breve.
Si la pregunta no es relevante, indícalo con amabilidad.

Contexto:
{context}

Pregunta:
{input}
""")

# Cadena RAG
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

def responder_pregunta(pregunta: str) -> str:
    respuesta = chain.invoke({
        "input": pregunta,
        "context": chunks
    })
    return respuesta
