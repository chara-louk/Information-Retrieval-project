from langchain.chains import LLMChain
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from ir import Index, Boolean

index = Index()
documents = index.get_docs()
tokenized_docs = index.tokenization(documents)
vocabulary = index.get_vocabulary()
boolean_inverted_index = index.inverted_index()

boolean_model = Boolean(boolean_inverted_index)

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
    words = doc_content.split("\n")
    return " ".join(words)


def query_handler(query, max_docs=5):
    relevant_doc_ids = boolean_model.process_query(query)
    relevant_docs_info = []

    for doc_id in relevant_doc_ids[:max_docs]:
        doc_content = next((content for doc_name, content in documents if doc_name == doc_id), None)
        if doc_content:
            reconstructed_content = reconstruct_document(doc_content)
            relevant_docs_info.append((doc_id, reconstructed_content))

    return relevant_docs_info

while True:
    user_query = input("Ask a question: ").strip()

    try:
        ir_query = user_query
        relevant_docs_info = query_handler(ir_query, max_docs=5)
        document_contents = "\n\n".join([content for doc_id, content in relevant_docs_info])

        if document_contents:
            combined_input = f"Question: {ir_query}\n\nDocuments:\n{document_contents}\n\nPlease provide a detailed answer based on the documents."
        else:
            combined_input = f"Question: {ir_query}\n\nNo relevant documents found."

        # AFTA EINAI APLA YA DEBUGGING
        print(f"Query: {ir_query}")
        print(f"Documents: {document_contents}")
        print(f"Combined input being passed to model: {combined_input}")

        memory = ConversationBufferMemory()

        chat_chain = LLMChain(
            llm=chat_model,
            prompt=chat_prompt_template,
            memory=memory,
            verbose=True
        )

        model_inputs = {"combined_input": combined_input}
        response = chat_chain.invoke(model_inputs)

        if response:
            print(f"Chatbot: {response['text']}")
        else:
            print("Chatbot did not respond.")
    except Exception as e:
        print(f"Error: {e}")

    cont = input("Would you like to ask another question? (yes/no): ").strip().lower()
    if cont != 'yes':
        print("Goodbye!")
        break
