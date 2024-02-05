import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.text_splitter import CharacterTextSplitter
import openai

# Set OpenAI API Key
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Document file paths
file1 = "./data/DIVISION OF ASSETS AFTER DIVORCE.txt"
file2 = "./data/INHERITANCE.txt"

def openai_setting():
    embedding = OpenAIEmbeddings()
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    return embedding, llm

def law_content_splitter(path, splitter="CIVIL CODE"):
    with open(path) as f:
        law_content = f.read()
    law_content_by_article = law_content.split(splitter)[1:]
    text_splitter = CharacterTextSplitter()
    return text_splitter.create_documents(law_content_by_article)

def is_greeting(input_str):
    greetings = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "hi there", "hello there", "hey there", 
        "whats up", "ciao", "salve", "buongiorno", 
        "buona sera", "buonasera", "buon pomeriggio", "buonpomeriggio", 
        "come stai", "comestai", "come va", "comeva", "come sta", "comesta"
    ]
    return any(greet in input_str.lower() for greet in greetings)

def chatbot1(question):
    try:
        return agent.run(question)
    except:
        return "I'm sorry, I'm having trouble understanding your question."

def chatbot(input_str):
    if is_greeting(input_str):
        return "Hello! Ask me your question about Italian Divorce or Inheritance Law?"
    else:
        return chatbot1(input_str)


# ## If you wanna disable Greeting in the chatbot, use this code:
# def chatbot(input_str):
#     # Directly process every input as a question
#     response = chatbot1(input_str)
#     if response == "N/A":
#         return "I'm sorry, I'm having trouble understanding your question. Could you please rephrase it or provide more context"
#     else:
#         return response



# Splitting the content of law documents
divorce_splitted = law_content_splitter(file1)
inheritance_splitted = law_content_splitter(file2)

# Initializing embedding and language model
embedding, llm = openai_setting()

# Define the prompts
divorce_prompt = """As a specialized bot in divorce law, you should offer accurate insights on Italian divorce regulations.
You should always cite the article numbers you reference. 
Ensure you provide detailed and exact data. 
If a query doesn't pertain to the legal documents, you should remind the user that it falls outside your expertise.
You should be adept at discussing the various Italian divorce categories, including fault-based divorce, mutual-consent divorce, and divorce due to infidelity.
You should guide users through the prerequisites and procedures of each divorce type, detailing the essential paperwork, expected duration, and potential legal repercussions. 
You should capably address queries regarding asset allocation, child custody, spousal support, and other financial concerns related to divorce, all while staying true to Italian legislation.
{context}

Question: {question}"""
DIVORCE_BOT_PROMPT = PromptTemplate(template=divorce_prompt, input_variables=["context", "question"])

# define inheritance prompt
inheritance_prompt = """As a specialist in Italian inheritance law, you should deliver detailed and accurate insights about inheritance regulations in Italy.
You should always cite the article numbers you reference. 
When responding to user queries, you should always base your answers on the provided context.
Always MUST MUST cite the specific article numbers you mention and refrain from speculating.
Maintain precision in all your responses.
If a user's question doesn't align with the legal documents, you should point out that it's beyond your domain of expertise.
You should elucidate Italian inheritance law comprehensively, touching on topics such as testamentary inheritance, intestate inheritance, and other pertinent subjects.
Make sure to elaborate on the obligations and rights of inheritors, the methodology of estate distribution, asset assessment, and settling debts, all while adhering to Italian law specifics.
You should adeptly tackle questions about various will forms like holographic or notarial wills, ensuring you clarify their legitimacy within Italian jurisdiction. 
Offer advice on creating a will, naming heirs, and managing potential conflicts.
You should provide detailed information on tax nuances associated with inheritance in Italy, inclusive of exemptions, tax rates, and mandatory disclosures.

{context}

Question: {question}"""
INHERITANCE_BOT_PROMPT = PromptTemplate(template=inheritance_prompt, input_variables=["context", "question"])

# Setup for Chroma databases and RetrievalQA
chroma_directory = "./docs/chroma"

inheritance_db = Chroma.from_documents(
    documents=inheritance_splitted,
    embedding=embedding,
    persist_directory=chroma_directory,
)
inheritance = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=inheritance_db.as_retriever(),
    chain_type_kwargs={"prompt": INHERITANCE_BOT_PROMPT},
)

divorce_db = Chroma.from_documents(
    documents=divorce_splitted, embedding=embedding, persist_directory=chroma_directory
)
divorce = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=divorce_db.as_retriever(),
    chain_type_kwargs={"prompt": DIVORCE_BOT_PROMPT},
)

# Define the tools for the chatbot
tools = [
    Tool(
        name="Divorce Italian law QA System",
        func=divorce.run,
        description="useful for when you need to answer questions about divorce laws in Italy.Give also the number of article you use for it.",
    ),
    Tool(
        name="Inheritance Italian law QA System",
        func=inheritance.run,
        description="useful for when you need to answer questions about inheritance laws in Italy.Give also the number of article you use for it.",
    ),
]

# Initialize conversation memory and ReAct agent
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key="output")
react = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent = AgentExecutor.from_agent_and_tools(tools=tools, agent=react.agent, memory=memory, verbose=False)





# Streamlit UI Setup
def setup_ui():
    st.set_page_config(page_title="Italian Law Chatbot", page_icon="⚖️")
    st.title("🏛️ Legal Chatbot: Divorce and Inheritance Laws ")


    st.write("""
    [![HuggingFace Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/sattari/legal-chat-bot/tree/main)
    [![SATTARI.org](https://img.shields.io/badge/SATTARI.org-gray?logo=internetexplorer)](https://www.sattari.org)
    ![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fsattari-legal-chat-bot.hf.space&label=Visitors&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
    """)

    
    
    st.info(
        "Check out full tutorial to build this app on Streamlit [📝 blog](https://sattari.org/legal-chatbot-divorce-and-inheritance-italy-laws/)",
        icon="ℹ️",
    )


    st.success(
        "Check out [Prompt Examples List](https://huggingface.co/spaces/sattari/legal-chat-bot/blob/main/promptExamples.txt) to know how to interact with this ChatBot 🤗 ",
        icon="✅",
    )
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm here to help you with Italian Divorce or Inheritance Law.",
            }
        ]

    # Display previous messages and handle new user input
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question about Italian Divorce or Inheritance Law:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display chatbot response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response = chatbot(user_input)  # Your existing chatbot function
            response_placeholder.markdown(response)

        # Append the response to the conversation history
        st.session_state.messages.append({"role": "assistant", "content": response})


## Made by Pouya /  www.SATTARI.org

    
if __name__ == "__main__":
    setup_ui()
