from langchain.chains import LLMChain
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from ir import Index, VSM

index = Index()
documents = index.get_docs()
tokenized_docs = index.tokenization(documents)
vocabulary = index.get_vocabulary()
inverted_weigh_ind = index.weighted_inverted_index()
total_docs = len(tokenized_docs)

vsm_model = VSM(inverted_weigh_ind, vocabulary, total_docs)

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{combined_input}"),
    ("assistant", "Based on the provided documents, here is my response.")
])

try:
    chat_model = ChatOllama(model="llama3.2:latest", api_key="")
except Exception as e:
    print(f"Failed to initialize ChatOllama: {e}")
    exit()


def reconstruct_document(doc_content):
    return " ".join(doc_content.split("\n"))


def query_handler(query, max_docs=5):
    try:
        ranked_docs = vsm_model.rank_documents(query)
        relevant_doc_ids = [doc_id for doc_id, _ in ranked_docs[:max_docs]]
        relevant_docs_info = [
            (doc_id, reconstruct_document(content))
            for doc_id in relevant_doc_ids
            for doc_name, content in documents
            if str(doc_name) == str(doc_id)
        ]
        return relevant_docs_info
    except Exception as e:
        print(f"Error in query_handler: {e}")
        return []


while True:
    user_query = input("Ask a question: ").strip()

    try:
        relevant_docs_info = query_handler(user_query, max_docs=5)
        document_contents = "\n\n".join([content for _, content in relevant_docs_info])

        if document_contents:
            combined_input = (
                f"Question: {user_query}\n\nDocuments:\n{document_contents}\n\n"
                "Please provide a detailed answer based on the documents."
            )
        else:
            combined_input = f"Question: {user_query}\n\nNo relevant documents found."

        # for debugging!
        print(f"Query: {user_query}")
        print(f"Documents: {document_contents}")
        print(f"Combined input being passed to model: {combined_input}")

        memory = ConversationBufferMemory()

        chat_chain = LLMChain(
            llm=chat_model,
            prompt=chat_prompt_template,
            memory=memory,
            verbose=True
        )
        response = chat_chain.invoke({"combined_input": combined_input})

        if response and response.get("text"):
            print(f"Chatbot: {response['text']}")
        else:
            print("Chatbot did not respond.")

    except Exception as e:
        print(f"Error: {e}")

    cont = input("Would you like to ask another question? (yes/no): ").strip().lower()
    if cont != 'yes':
        print("Goodbye!")
        break
