import os
from pathlib import Path
# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import (
    CSVLoader, 
    PDFMinerLoader, 
    TextLoader, 
    UnstructuredExcelLoader,
    Docx2txtLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, 
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredImageLoader,
)

# load_dotenv()
#ROOT_DIRECTORY = str(Path(__file__).resolve().parent.parent)
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing documents
SOURCE_DIRECTORY_NAME = "SOURCES"
FULL_PATH_TO_SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY,SOURCE_DIRECTORY_NAME)

# Define the folder for storing database
DB_DIRECTORY_NAME = "DB"
PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY,DB_DIRECTORY_NAME)

# Can be changed to a specific number
MAX_CPUS = os.cpu_count() or 8

# Can be changed to a specific number
MAX_THREADS_PER_CPU = 8

# Can be changed to a specific number
MAX_FILES_PER_THREAD = 20
MAX_IMAGE_FILES_PER_THREAD = 4

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)
# https://python.langchain.com/docs/modules/data_connection/document_loaders

DOCUMENT_MAP = {
    ".txt":  TextLoader,
    ".md":   TextLoader,
    #".py":   TextLoader,
    #".yml":  TextLoader,
    ".pdf":  PDFMinerLoader,
    ".csv":  CSVLoader,
    ".xls":  UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xlsm": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    #".doc":  UnstructuredWordDocumentLoader, # requires libreoffice pacakages
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt":  UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    #".json": JSONLoader,
    ".jpg":  UnstructuredImageLoader, # requires OCR packages from pytesseract
    ".jpeg": UnstructuredImageLoader,
    ".png":  UnstructuredImageLoader,
}

IMAGE_FILE_TYPE_LIST = [".jpg",".jpeg",".png"]

# Default scanned documents csv file name
SCANNING_RECORD_CSV_FILE_NAME = 'MyDocumentsIndex.csv'

# Default conversation history file name
CONVERSATION_HISTORY_FILE_NAME = "MyConversationHistory.csv"

# Default Instructor Model
#EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"


#### SELECT AN OPEN SOURCE LLM (LARGE LANGUAGE MODEL)
    # Select the Model ID and model_basename
    # load the LLM for generating Natural Language responses

#MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
#MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

####
#### (FOR HF MODELS)
####

# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

# for GPTQ (quantized) models
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
#MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
#MODEL_BASENAME = "model.safetensors" # Requires
# ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
#MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
#MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"

#MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
#MODEL_BASENAME = "model.safetensors"

#MODEL_ID = "TheBloke/falcon-40b-instruct-GPTQ"
#MODEL_BASENAME = "gptq_model-4bit--1g.safetensors"

#MODEL_ID = "TheBloke/Llama-2-13B-chat-GPTQ"
#MODEL_BASENAME = "model.safetensors"

#MODEL_ID = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
#MODEL_BASENAME = "model.safetensors"

#MODEL_ID = "TheBloke/stable-vicuna-13B-GPTQ"
#MODEL_BASENAME = "stable-vicuna-13B-GPTQ-4bit.compat.no-act-order.safetensors"

MODEL_ID = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
MODEL_BASENAME = "model.safetensors"
