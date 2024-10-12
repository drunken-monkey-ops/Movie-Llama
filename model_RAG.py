from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




persist_directory= 'chroma_vector_store'

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
loaded_vectorstore = Chroma(
    embedding_function= local_embeddings,
    persist_directory=persist_directory
)

question = "I'm looking for an Action comedy films"

docs = loaded_vectorstore.similarity_search(question)


print(docs)


model = ChatOllama(
    model="llama3.1:latest",
)


RAG_TEMPLATE = """
You are an assistant for recommanding movies. Use the following pieces of retrieved context to suggest atleast 3 movies. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{query}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

query = "I am want to burst my stoach out I would like to watch some good comedy films"

docs = loaded_vectorstore.similarity_search(query)

print(chain.invoke({"context": docs, "query": query}))