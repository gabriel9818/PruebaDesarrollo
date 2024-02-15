import time
import datetime
# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo
import os
# Importa la biblioteca Streamlit para crear aplicaciones web interactivas
import streamlit as st
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# Importa FAISS del módulo langchain para realizar búsqueda de similitud
from langchain import FAISS
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain.llms import OpenAI
# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain.callbacks import get_openai_callback
# Importa el módulo langchain
import langchain

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un archivo HTML
def process_html(html_content):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(html_content)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

# Función para procesar el texto extraído de un archivo RSS
def process_rss(rss_content):
    # Aquí puedes agregar la lógica para procesar el contenido del archivo RSS
    # Por simplicidad, este ejemplo simplemente utilizará el contenido RSS directamente

    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(rss_content)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

# Función principal de la aplicación
def main():
    st.title("Preguntas a archivos HTML o RSS")

    html_file = st.file_uploader("Sube tu archivo HTML", type="html")
    rss_file = st.file_uploader("Sube tu archivo RSS", type="rss")

    if html_file is not None:
        html_content = html_file.read().decode("utf-8")
        knowledgeBase = process_html(html_content)
    elif rss_file is not None:
        rss_content = rss_file.read().decode("utf-8")
        knowledgeBase = process_rss(rss_content)
    else:
        st.warning("Por favor, sube un archivo HTML o RSS.")

    if html_file is not None or rss_file is not None:
        query = st.text_input('Escribe tu pregunta para el archivo HTML o RSS...')

        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()  # Detiene la ejecución de la aplicación

        if query:
            start_time = time.time()  # Obtiene el tiempo de inicio de la consulta
            docs = knowledgeBase.similarity_search(query)
            execution_time = time.time() - start_time  # Calcula el tiempo de ejecución

            model = "gpt-3.5-turbo-instruct"  # Acepta 4096 tokens
            temperature = 0  # Valores entre 0 - 1

            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                cost_value = cost  # Obtiene el costo de la operación

                st.write(response["output_text"])

            st.write(f"Tiempo de ejecución: {datetime.timedelta(seconds=execution_time)}")  # Muestra el tiempo de ejecución
            st.write(f"Costo: {cost_value}")  # Muestra el costo

if __name__ == "__main__":
    main()
