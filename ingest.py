import ast
import logging
import os
import datetime
import hashlib
import csv

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import tempfile

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    IMAGE_FILE_TYPE_LIST,
    EMBEDDING_MODEL_NAME,
    MAX_FILES_PER_THREAD,
    MAX_CPUS,
    MAX_IMAGE_FILES_PER_THREAD,
    MAX_THREADS_PER_CPU,
    PERSIST_DIRECTORY,
    FULL_PATH_TO_SOURCE_DIRECTORY,
    ROOT_DIRECTORY,
    SCANNING_RECORD_CSV_FILE_NAME,
)


import pytesseract 
pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract-OCR\\tesseract.exe'

os.environ['CURL_CA_BUNDLE'] = ''





# store scanning history
records = {}

# store scanned files in memory
scan_files = []
scan_image_files =[]

def add_document_record(originalPath, fileName, is_image_type):
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
            "is_image": is_image_type,
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
       logging.error(f"Error: File '{filePath}' not found.")
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

# file must not be 0 byte
# file must be with the valid file type
# file must be unique in the whole dictionary
def is_existing_record(filePath):
    return get_record(filePath) != None

def load_single_document(file_path: str) -> Document:
    doc = None
    try:
        # Loads a single document from a file path
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
        else:
            raise ValueError("Document type is undefined")
        doc = loader.load()[0]
    except Exception as ex:
        logging.error(f"While scanning file: {file_path}, following error occurred: "+str(ex))
        raise ex
    return doc

# open multiple therads for task paralle running
def load_document_batch(files:list, number_of_files):
    logging.info("Loading document batch ...")
    number_of_threads = min(MAX_THREADS_PER_CPU, max(number_of_files, 1))
    files_per_batch = min(MAX_FILES_PER_THREAD, round(number_of_files / number_of_threads))
    
    data_list = []
    
    # create a thread pool
    with ThreadPoolExecutor(max_workers = number_of_threads) as executor:
        thread_futures = []
        # split file list into smaller batch with the batch size aligned with nnumber of threads
        
        for i in range(0, number_of_files, files_per_batch):
            # select a chunk of filenames
            file_segements = files[i : (i + files_per_batch)]
        
            # load files
            #futures = [executor.submit(load_single_document, name) for name in files]
            # submit the task
            #thread_futures.append([executor.submit(load_single_document, name) for name in file_segements])
        
            # submit the tasks for file segements and execute them
            for result in executor.map(load_single_document, file_segements,timeout=600):
                data_list.append(result)
        #for future in as_completed(thread_futures):
            # collect data
         #   data_list.append(future.result())
        # return data and file paths
        return (data_list, files)
        

def load_image_document_batch(files:list,number_of_files):
    logging.info("Loading image document batch ...")
    number_of_threads = min(MAX_THREADS_PER_CPU, max(number_of_files, 1))
    files_per_batch = min(MAX_IMAGE_FILES_PER_THREAD, round(number_of_files / number_of_threads))
    
    data_list = []
    
    # create a thread pool
    with ThreadPoolExecutor(max_workers = number_of_threads) as executor:
        thread_futures = []
        # split file list into smaller batch with the batch size aligned with nnumber of threads
        
        for i in range(0, number_of_files, files_per_batch):
            # select a chunk of filenames
            file_segements = files[i : (i + files_per_batch)]
                    
            # submit the tasks for file segements and execute them
            for result in executor.map(load_single_document, file_segements, timeout=300):
                data_list.append(result)
        
            #tasks = [executor.submit(load_single_document, name) for name in file_segements]
            
        # collect data
        #data_list = [future.result() for future in thread_futures]
        #for future in as_completed(thread_futures):
            # collect data
         #   data_list.append(future.result())
        
        # return data and file paths
        return (data_list, files)

def execute_task(numberOfFiles, scan_file_list, func, records):
    n_workers = min(MAX_CPUS, max(numberOfFiles, 1))
    files_per_batch = round(numberOfFiles / n_workers)
    documents = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        
        for i in range(0, numberOfFiles, files_per_batch):
            # select a chunk of filenames
            file_segements = scan_file_list[i : (i + files_per_batch)]
            # submit the task
            future = executor.submit(func, file_segements,numberOfFiles)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                documents.extend(contents)
                for content in contents:
                    docPath = content.metadata['source']
                    if(is_existing_record(docPath)):
                        records.get(generate_record_id(docPath))['scanned'] = True
            except Exception as ex:
                logging.error(f"While scanning files, following error was captured: " + str(ex))
                continue
    return documents



def search_files_recursive(source_dir: str, all_files: list[str]):
    for file_name in all_files:
        source_file_path = os.path.join(source_dir,file_name)
        
        # check if the path is a directory, then recursively dive into the sub directory first
        if os.path.isdir(source_file_path):
            all_files = os.listdir(source_file_path)
            search_files_recursive(source_file_path,all_files)
        
        # ignore any file starting with . or starting with $, they are not valid for processing
        if not file_name.startswith(".") \
           and not file_name.startswith("~$") :
            file_extension = os.path.splitext(file_name)[1]
            
            # only if file extension is supported and file size is not equal to zero
            # and only if record doesn't exist or an existing record has not yet been scanned
            is_zero_byte = os.path.getsize(source_file_path) == 0
            if file_extension in DOCUMENT_MAP.keys() and not is_zero_byte:
                isExistingRecord = is_existing_record(source_file_path)
                
                if not isExistingRecord \
                    or isExistingRecord \
                    and not get_record(source_file_path)['scanned']:
                        # add the file for scanning
                        if file_extension in IMAGE_FILE_TYPE_LIST:
                            scan_image_files.append(source_file_path)
                        else: 
                            scan_files.append(source_file_path)
                        
                        # if this file was not recorded before, then add it to record list
                        if not isExistingRecord:
                            add_document_record(source_dir, file_name, file_extension in IMAGE_FILE_TYPE_LIST)

def get_record(source_file_path):
    return records.get(generate_record_id(source_file_path))


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    search_files_recursive(source_dir, all_files)
    
    numberOfFiles = len(scan_files)
    numberOfImageFiles = len(scan_image_files)
    if numberOfFiles == 0 and numberOfImageFiles == 0:
        return []
    
    start = datetime.datetime.now()
    
    total_documents = []
    if numberOfFiles != 0:
        logging.info(f"{numberOfFiles} files have been found and will be loaded now ...\n")  
        # Have at least one worker and at most INGEST_THREADS workers
        loaded_doc_files = execute_task(numberOfFiles, scan_files, load_document_batch, records)
        #for doc in tmp_docs:
          #  total_documents.append(doc)
        total_documents.extend(loaded_doc_files)
    
    if numberOfImageFiles != 0:
        logging.info(f"{numberOfImageFiles} image files have been found and will be loaded now ...\n")
        # Have at least one worker and at most INGEST_THREADS workers
        loaded_image_files = execute_task(numberOfImageFiles, scan_image_files, load_image_document_batch, records)
        #for doc in tmp_docs:
         #   total_documents.append(doc)
        total_documents.extend(loaded_image_files)
        
    end = datetime.datetime.now()
    logging.info(f"Total time spent for loading files: {end-start}s \n")
    
    return total_documents #[x for x in text_documents] + [y for y in image_documents]


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs, json_yaml_docs = [], [], []
    for doc in documents:
        try: 
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            elif file_extension == ".yml" or file_extension == ".yaml" or file_extension == ".json":    
                json_yaml_docs.append(doc)
            else:
                text_docs.append(doc)
        except Exception as ex:
            logging.error(f"While splitting document: {doc}, following error occurred: " +str(ex))
            continue
    return text_docs, python_docs, json_yaml_docs


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
    open_scanned_file()
    # Load documents and split in chunks
    logging.info(f"Loading documents from {FULL_PATH_TO_SOURCE_DIRECTORY}")
    documents = load_documents(FULL_PATH_TO_SOURCE_DIRECTORY)
    
    if len(documents) == 0:
        logging.info("No more documents have been identifed for scanning. Exit now ...")
        return
    
    # save temporary files to local tempfile
    #with tempfile.NamedTemporaryFile(dir=ROOT_DIRECTORY) as tmp:
        #for document in documents:
          #  new_var = pickle.dumps(document)
         #   tmp.write(new_var)
        #    tmp.write(b'\n')  # add \n at the end of each document
        
       # print(tmp.name+"\n")
      #  tmp.flush()
     #   tmp.close()
    
    
    logging.info(f"\n\n========= {len(documents)} documents have been successfully scanned from {FULL_PATH_TO_SOURCE_DIRECTORY} =========\n\n")

    text_documents, python_documents, yaml_json_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # use OPENAI fast tokenizer
    #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
     #   model_name = "gpt-3.5-turbo",
      #  allowed_special="all",
       # separators=["\n\n", "\n", ".", ","],
        #chunk_size = 100,
        #chunk_overlap=20
    #)
    
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    
    logging.info(f"Loaded {len(documents)} documents from {FULL_PATH_TO_SOURCE_DIRECTORY}")
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

    save_scanned_files(records)


def open_scanned_file():
    if(os.path.exists(SCANNING_RECORD_CSV_FILE_NAME)):
        try:
            with open(SCANNING_RECORD_CSV_FILE_NAME,'r', encoding='utf-8') as dumpFile:
                oldRecords = csv.DictReader(dumpFile, fieldnames=('hashcode','tuple'))
                num = 0
                unscanned = 0
                for row in oldRecords:
                    currentRecord = ast.literal_eval(row['tuple'])
                    records[row['hashcode']] = {
                        "original_path": currentRecord['original_path'],
                        "file_name": currentRecord['file_name'],
                        "timestamp": currentRecord['timestamp'],
                        "is_image": currentRecord['is_image'],
                        "scanned": currentRecord['scanned'],
                    }
                    if currentRecord['scanned'] == False:
                        unscanned = unscanned+1
                    num=num+1
                logging.info(f"{num} records are loaded from {SCANNING_RECORD_CSV_FILE_NAME} and " + f"{unscanned} files are not scanned from last run.")    
        except Exception as ex:
            logging.error(f"While opening scan file from {SCANNING_RECORD_CSV_FILE_NAME}, following error occurred: " + str(ex))      

def save_scanned_files(records:dict):
    if os.path.exists(SCANNING_RECORD_CSV_FILE_NAME):
        try:
            os.remove(SCANNING_RECORD_CSV_FILE_NAME)
        except Exception as ex:
            logging.error(f"While saving scan file from {SCANNING_RECORD_CSV_FILE_NAME}, following error occurred: " + str(ex))
            
    with open(SCANNING_RECORD_CSV_FILE_NAME,'a+', encoding='utf-8') as dumpFile:
        writer = csv.writer(dumpFile)
        writer.writerows(records.items())

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
    