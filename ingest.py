import ast
import logging
import os
import datetime
import hashlib
import csv

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    DUMP_CSV_FILE_NAME,
)

# store scanned files
records = {}

def add_document_record(originalPath, fileName):
    if(os.path.isdir(os.path.join(originalPath,fileName))):
        return
    
    hashcode = generate_record_id(os.path.join(originalPath,fileName))
    if hashcode in records :
        print(f"ID {hashcode} already exists and file has been scanned.")
    else:
        records[hashcode] = {
            "original_path": originalPath,
            "file_name": fileName,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "scanned": False,
        }
        #print(f"New record {hashcode} added successfully.")

BUF_SIZE = 8192  # lets read stuff in 64kb chunks!

def generate_record_id(filePath):
    try:
        hasher = hashlib.new('sha256')
        with open(filePath, 'rb') as file:
            while chunk := file.read(BUF_SIZE):
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
       print(f"Error: File '{filePath}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        return None

# file must not be 0 byte
# file must be with the valid file type
# file must be unique in the whole dictionary
def is_existing_record(filePath):
    return get_record(filePath) != None

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


scan_files = []
def load_documents_recursive(source_dir: str, all_files: list[str]):
    for file_name in all_files:
        source_file_path = os.path.join(source_dir,file_name)
        # check if the path is a directory, then recursively dive into the sub directory first
        if os.path.isdir(source_file_path):
            all_files = os.listdir(source_file_path)
            load_documents_recursive(source_file_path,all_files)
        # ignore any file starting with ., they are not valid for processing
        if not file_name.startswith('.'):
            file_extension = os.path.splitext(file_name)[1]
            
            # only if file extension is supported and file size is not equal to zero
            # and only if record doesn't exist or an existing record has not yet been scanned
            isZeroByte = os.path.getsize(source_file_path) == 0
            if file_extension in DOCUMENT_MAP.keys() and not isZeroByte:
                isExistingRecord = is_existing_record(source_file_path)
                
                if not isExistingRecord \
                    or isExistingRecord and not get_record(source_file_path)['scanned']:
                        # add the file for scanning
                        scan_files.append(source_file_path)
                        if not isExistingRecord:
                            add_document_record(source_dir, file_name)

def get_record(source_file_path):
    return records.get(generate_record_id(source_file_path))


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    load_documents_recursive(source_dir, all_files)
    
    numberOfFiles = len(scan_files)
    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(numberOfFiles, 1))
    chunksize = 2 # round(numberOfFiles / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        
        for i in range(0, numberOfFiles, chunksize):
            # select a chunk of filenames
            filepaths = scan_files[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
                for content in contents:
                    docPath = content.metadata['source']
                    if(is_existing_record(docPath)):
                        records.get(generate_record_id(docPath))['scanned'] = True
            except Exception as ex:
                print(str(ex))
                continue
    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    open_dump_csv()
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    
    if len(documents) == 0:
        print("No more documents have been identifed for scanning. Exit now ...")
        return
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY} for scanning")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    #embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None

    save_dump_csv(records)


def open_dump_csv():
    if(os.path.exists(DUMP_CSV_FILE_NAME)):
        try:
            with open(DUMP_CSV_FILE_NAME,'r') as dumpFile:
                oldRecords = csv.DictReader(dumpFile, fieldnames=('hashcode','tuple'))
                num = 0
                for row in oldRecords:
                    currentRecord = ast.literal_eval(row['tuple'])
                    records[row['hashcode']] = {
                        "original_path": currentRecord['original_path'],
                        "file_name": currentRecord['file_name'],
                        "timestamp": currentRecord['timestamp'],
                        "scanned": currentRecord['scanned'],
                       # "processed": currentRecord['processed']
                    }
                    num=num+1
                print(f"{num} records loaded from scanning csv file")    
        except Exception as ex:
            print(str(ex))          

def save_dump_csv(records):
    if os.path.exists(DUMP_CSV_FILE_NAME):
        try:
            os.remove(DUMP_CSV_FILE_NAME)
        except Exception as ex:
            print(str(ex))
            
    with open(DUMP_CSV_FILE_NAME,'a+') as dumpFile:
        writer = csv.writer(dumpFile)
        writer.writerows(records.items())

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
    