import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage


class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.vectorstore = None
        self.retriever = None
        self.chat_history = []  # stores HumanMessage / AIMessage objects
        self.doc_filename = ""

        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0.2,
            openai_api_key=api_key
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # ── Document Loading ───────────────────────────────────────────
    def load_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        self.doc_filename = os.path.basename(file_path)

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif ext in [".pptx", ".ppt"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        pages = loader.load()

        # Tag each chunk with its page number for citations
        for i, page in enumerate(pages):
            if "page" not in page.metadata:
                page.metadata["page"] = i + 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = splitter.split_documents(pages)

        if not chunks:
            raise ValueError("No text could be extracted from this file.")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 12}
        )
        self.chat_history = []  # reset history on new doc

    # ── Rephrase question using chat history ───────────────────────
    def _rephrase_question(self, question: str) -> str:
        """If there's chat history, rephrase the question to be standalone."""
        if not self.chat_history:
            return question

        history_text = ""
        for msg in self.chat_history[-6:]:  # last 3 exchanges
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # truncate long AI messages for the rephrase prompt
                content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                history_text += f"Assistant: {content}\n"

        rephrase_prompt = f"""Given this conversation history and a follow-up question, rephrase the follow-up into a standalone question that captures the full context needed to search a document.

Conversation:
{history_text}
Follow-up question: {question}

Standalone question (just output the question, nothing else):"""

        rephrased = self.llm.invoke(rephrase_prompt).content.strip()
        return rephrased

    # ── Ask with memory + citations ────────────────────────────────
    def ask(self, question: str) -> dict:
        """
        Returns dict with:
          - answer: str
          - sources: list of {page, snippet}
        """
        if not self.retriever:
            return {"answer": "No document loaded.", "sources": []}

        # Rephrase using chat history so follow-ups work properly
        standalone_q = self._rephrase_question(question)

        # Retrieve relevant chunks
        docs = self.retriever.invoke(standalone_q)

        # Build context with page markers
        context_parts = []
        for doc in docs:
            page_num = doc.metadata.get("page", "?")
            context_parts.append(f"[Page {page_num}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # Build chat history string for the prompt
        history_str = ""
        for msg in self.chat_history[-6:]:
            if isinstance(msg, HumanMessage):
                history_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                truncated = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
                history_str += f"Assistant: {truncated}\n"

        prompt = f"""You are a helpful academic assistant with memory of the conversation.

Previous conversation:
{history_str if history_str else "None"}

Document context (with page numbers):
{context}

Current question: {question}

Instructions:
- Answer thoroughly using the document context
- If this is a follow-up question, build on the previous conversation
- For follow-ups like "explain more", "in detail", "give examples" — expand fully
- At the end of your answer, add a line: SOURCES: [list the page numbers you used, e.g. Page 2, Page 5]
- Only say you couldn't find something if the topic is truly absent

Answer:"""

        response = self.llm.invoke(prompt).content

        # Parse sources from response
        sources = []
        answer = response
        if "SOURCES:" in response:
            parts = response.rsplit("SOURCES:", 1)
            answer = parts[0].strip()
            source_line = parts[1].strip()
            # Extract page numbers
            import re
            page_nums = re.findall(r'[Pp]age\s*(\d+)', source_line)
            seen = set()
            for pg in page_nums:
                pg_int = int(pg)
                if pg_int not in seen:
                    seen.add(pg_int)
                    # Find a snippet from that page
                    snippet = ""
                    for doc in docs:
                        if doc.metadata.get("page") == pg_int:
                            snippet = doc.page_content[:120].replace("\n", " ").strip() + "..."
                            break
                    sources.append({"page": pg_int, "snippet": snippet})

        # If no sources parsed, fall back to doc metadata
        if not sources:
            seen = set()
            for doc in docs[:3]:
                pg = doc.metadata.get("page", "?")
                if pg not in seen:
                    seen.add(pg)
                    snippet = doc.page_content[:120].replace("\n", " ").strip() + "..."
                    sources.append({"page": pg, "snippet": snippet})

        # Save to chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        # Keep history from growing too large
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return {"answer": answer, "sources": sources}

    def clear_history(self):
        self.chat_history = []

    # ── Summarize ──────────────────────────────────────────────────
    def summarize(self, detail_level: str = "Medium (1-2 paragraphs)") -> str:
        if not self.vectorstore:
            return "No document loaded."

        level_map = {
            "Short (3-4 sentences)": "in 3-4 sentences covering the main idea",
            "Medium (1-2 paragraphs)": "in 1-2 well-structured paragraphs",
            "Detailed (full breakdown)": "in detail with clearly labeled sections: Overview, Key Topics, Main Findings, and Conclusion",
        }
        instruction = level_map.get(detail_level, "in 1-2 paragraphs")

        docs = self.vectorstore.similarity_search(
            "overview introduction topics findings conclusion methodology results", k=10
        )
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""You are an academic assistant. Summarize this document {instruction}.
Be specific — mention actual topics, methods, findings, and conclusions from the document. Do not be vague.

Document content:
{context}

Summary:"""
        return self.llm.invoke(prompt).content

    # ── Quiz ───────────────────────────────────────────────────────
    def generate_quiz(self, quiz_type: str = "Multiple Choice (MCQ)", num_questions: int = 5) -> str:
        if not self.vectorstore:
            return "No document loaded."

        docs = self.vectorstore.similarity_search(
            "key concepts definitions facts important findings results methods", k=8
        )
        context = "\n\n".join(d.page_content for d in docs)

        type_map = {
            "Multiple Choice (MCQ)": "multiple choice questions. Each must have 4 options labeled A, B, C, D on separate lines. End each question with 'Answer: X'",
            "True / False": "True/False questions. After each question write 'Answer: True/False' and one sentence explanation.",
            "Mixed": "a mix of multiple choice (with A/B/C/D options) and True/False questions. Clearly label each type.",
        }
        q_format = type_map.get(quiz_type, type_map["Multiple Choice (MCQ)"])

        prompt = f"""You are an academic quiz generator. Create {num_questions} {q_format}

Requirements:
- Base every question on the document content below
- Number each question (1., 2., 3. ...)
- Mix easy and challenging questions
- Test real understanding not just definitions

Document content:
{context}

Quiz:"""
        return self.llm.invoke(prompt).content