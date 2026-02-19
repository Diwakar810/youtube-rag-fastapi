from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from re import search
from langchain_classic.retrievers import MultiQueryRetriever




load_dotenv() # loading the key from the .env file



# llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.1)

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
parser=StrOutputParser()


def get_script(url:str):
    video_id = url
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        # print(transcript_list)  # This will print FetchedTranscript object info
        transcript = " ".join(chunk.text for chunk in transcript_list.snippets)

        # print(transcript)

    except TranscriptsDisabled:
        print("Transcript is disabled for this video")
    except Exception as e:  # Fixed: Added 'as' keyword & proper spacing
        print(f"There is an Exception: {e}")  # Fixed: Use f-string

    return transcript

# creating the cunks of the youtube transcript and reutrn the chunks docks

def creat_chunks(transcript:str):
    docs=splitter.create_documents([transcript])
    return docs

# transcript=get_script("YFjfBk8HI5o")
# docs=creat_chunks(transcript)

#sotre the docs in the faiss database
    # vector_store=FAISS.from_documents(documents=docs,embedding=embedding)

# get the query

# query=input("enter the query")

#retrieve the relevant documnet from the vectore store related to the query using multi query serach
# creating the basse retriever multiQueryRetreiver

# base_retriever=MultiQueryRetriever.from_llm(
#     retriever=vector_store.as_retriever(search_kwargs={"k":5}),
#     llm=llm
# )


# # create the compressor
# compressor=LLMChainExtractor.from_llm(llm=llm)
# compressor_retriever=ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=base_retriever
# )

# #context
# context=compressor_retriever.invoke(query)

# base_retriever=vector_store.as_retriever(
#     search_kwargs={"k":5}
# )
# docs1=base_retriever.invoke(query)
# context = " ".join(doc.page_content for doc in docs1)

prompt=PromptTemplate(
    template="""You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.
                {context}
                Question: {question}
                    """,
    input_variables=["context","question"]
)

#creating chain
# paraller_cahin=RunnableParallel({
#     "context":RunnableLambda(get_script()) |
# }
# )
# chain=prompt|llm|parser

# result=chain.invoke({"context":context,"question":query})
# print(result)

def ask_question(video_url,que):
    transcript=get_script(video_url)
    docs=creat_chunks(transcript)
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    task="text-generation")
    model=ChatHuggingFace(llm=llm)
    vector_store=FAISS.from_documents(documents=docs,embedding=embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs1 = retriever.invoke(que)
    context = "\n\n".join([doc.page_content for doc in docs1])
    
    chain = prompt | model | parser
    result = chain.invoke({"context": context, "question": que})

    return result





# print(ask_question("YFjfBk8HI5o","what is this video about"))






