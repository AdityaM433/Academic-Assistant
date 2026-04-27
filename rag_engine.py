from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.vectorstore = None
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    def load_document(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def ask(self, question: str) -> str:
        if not self.retriever:
            return "No document loaded."

        prompt = PromptTemplate.from_template("""You are a helpful academic assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:""")

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    def summarize(self, detail_level: str = "Medium (1–2 paragraphs)") -> str:
        if not self.vectorstore:
            return "No document loaded."
        level_map = {
            "Short (3–4 sentences)": "in 3-4 sentences",
            "Medium (1–2 paragraphs)": "in 1-2 paragraphs",
            "Detailed (full breakdown)": "in detail with clear sections and bullet points",
        }
        instruction = level_map.get(detail_level, "in 1-2 paragraphs")
        docs = self.vectorstore.similarity_search("main topics key ideas summary overview", k=8)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"Summarize the following academic document {instruction}.\n\nContent:\n{context}\n\nSummary:"
        return self.llm.invoke(prompt).content

    def generate_quiz(self, quiz_type: str = "Multiple Choice (MCQ)", num_questions: int = 5) -> str:
        if not self.vectorstore:
            return "No document loaded."
        docs = self.vectorstore.similarity_search("key concepts facts definitions", k=6)
        context = "\n\n".join(d.page_content for d in docs)
        type_map = {
            "Multiple Choice (MCQ)": "multiple choice questions (4 options A/B/C/D, mark the correct answer)",
            "True / False": "True/False questions with correct answer and brief explanation",
            "Mixed": "a mix of multiple choice and True/False questions",
        }
        q_format = type_map.get(quiz_type, type_map["Multiple Choice (MCQ)"])
        prompt = f"""Create {num_questions} {q_format} based strictly on this content.
Number each question. For MCQs list options on separate lines and clearly mark the answer.

Content:
{context}

Quiz:"""
        return self.llm.invoke(prompt).content