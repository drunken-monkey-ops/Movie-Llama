from langchain.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document

loader = JSONLoader(file_path='Actual_data.json' , jq_schema= '.[] | {title, genres, original_language ,overview}', text_content=False)


documents = loader.load()

processed_documents = []

for doc in documents:
    content = doc.page_content  # This should be a dictionary
    # Verify content type
    if not isinstance(content, dict):
        import json
        content = json.loads(content)  # Parse if it's a string

    title = content.get('title', 'N/A')
    original_language = content.get('original_language', 'N/A')
    overview = content.get('overview', 'N/A')
    genres = content.get('genres', [])

    # Ensure genres is a list
    if not isinstance(genres, list):
        genres = [genres] if genres else []

    genres_str = ', '.join(genres)

    page_content = (
        f"Title: {title}\n"
        f"original_language: {original_language}\n"
        f"Genres: {genres_str}\n"
        f"Overview: {overview}"
    )
    new_doc = Document(
        page_content=page_content,
        metadata={'genres': genres_str}
    )

    processed_documents.append(new_doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

all_splits = text_splitter.split_documents(processed_documents)

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings , persist_directory= 'chroma_vector_store')
