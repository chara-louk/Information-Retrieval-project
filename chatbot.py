from langchain.chains import LLMChain
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from ir import Index, Boolean, VSM

index = Index()
documents = index.get_docs()
tokenized_docs = index.tokenization(documents)
vocabulary = index.get_vocabulary()
boolean_inverted_index = index.inverted_index()
weighted_inverted_index = index.weighted_inverted_index()

total_docs = len(tokenized_docs)
boolean_model = Boolean(boolean_inverted_index)
vsm_model = VSM(weighted_inverted_index, vocabulary, total_docs)


def query_handler(query, model_type):
    if model_type.lower() == "boolean":
        return boolean_model.process_query(query)
    elif model_type.lower() == "vsm":
        ranked_docs = vsm_model.rank_documents(query)
        return [doc_id for doc_id, _ in ranked_docs]
    else:
        return "Invalid model type."


try:
    chat_model = ChatOllama(model="llama3.2:latest", api_key="")
except Exception as e:
    print(f"Failed to initialize ChatOllama: {e}")
    exit()

prompt_template = PromptTemplate(
    input_variables=["query", "model_response"],
    template="""Question: {query}

    Answer: Based on the relevant documents: {model_response}

    Let's think step by step and provide a detailed answer."""
)

memory = ConversationBufferMemory()

chat_chain = LLMChain(
    llm=chat_model,
    prompt=prompt_template,
    memory=memory
)

print("Chatbot is running! Type 'exit' to stop.")
while True:
    user_query = input("Ask question: ").strip()
    if user_query.lower() == "exit":
        print("Goodbye!")
        break

    try:
        if user_query.startswith("model:"):
            _, model_type, ir_query = user_query.split(":", 2)
            model_type = model_type.strip()
            ir_query = ir_query.strip()

            relevant_docs = query_handler(ir_query, model_type=model_type)

            if isinstance(relevant_docs, list):
                if not relevant_docs:
                    model_response = "No relevant documents were found for your query."
                else:
                    model_response = f"The following documents are relevant to your query: {', '.join(relevant_docs)}."
            else:
                model_response = relevant_docs  

            print(f"Chatbot (Model-{model_type.capitalize()}): {model_response}")
        else:
            model_response = "Not applicable"

        response = chat_chain.invoke({
            "query": user_query,
            "model_response": model_response
        })

        if response:
            print(f"Chatbot: {response}")
        else:
            print("Chatbot did not respond.")

    except Exception as e:
        print(f"Error: {e}")
