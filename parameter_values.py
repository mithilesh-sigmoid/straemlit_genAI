import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


def get_parameters(api_key, query):
    os.environ["OPENAI_API_KEY"] = api_key
    # Load data from a text file
    loader = TextLoader("knowledge_base.txt")  # Replace with your knowledge base
    documents = loader.load()

    # Split data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create a vector store with embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    # Initialize a retriever for querying the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_k=3)

    from langchain.tools import Tool
    import json

    def get_consolidation(info_str):
        
        info_dict= eval(info_str)
        
        return info_dict

    # Define the tool
    cost_consolidation_tool = Tool(
        name="Cost consolidation",
        func=get_consolidation,
        description="""You are an AI assistant tasked with analyzing questions and based on that give value of certain variables:
    list of variables are below:
    start_date:
    end_date:
    group_method:
    all_post_code: 
    all_customers:
    selected_postcodes: 
    selected_customers:

    I will provide you a question to answer, based on the question you need to provide variable values .

    Here are the questions I would like you to answer:
    1. How can I optimize the shipment costs for user ALLOGA UK (493).
    2. Can you optimize costs for shipments to zip code NG (313) between January and March 2024?

    To answer this, first think through your approach
    To answer this question, 
    1. You will need to find the start and end date first if it is not mentioned then start date will be df['SHIPPED_DATE'].min() and end date will be df['SHIPPED_DATE'].max()
    2. Determine the group_method, whether it 'Customer Level' or 'Post Code Level'
    3.  Determine the list of post codes or list of users that are mentioned in the question, if there is no mention of post code or users , then make all_post_code = False  if group method is Post Code Level otherwise keep it None, and  all_customers = False if group method is Customer Level otherwise keep it None.
    4. if there is a mention of certain users or zip codes, make a list of that.

    then return the value of all the required variables based on the questions.

    for example for the first question "How can I optimize the shipment costs for user ALLOGA UK (493)." the response should be similar to this but in dictionary format:
    start_date: df['SHIPPED_DATE'].min()
    end_date:df['SHIPPED_DATE'].max()
    group_method: 'Customer Level'
    all_post_code: None
    all_customers: False
    selected_postcodes: []
    selected_customers:  [ALLOGA UK (493)]

    for the 2nd question "Can you optimize costs for shipments to zip code NG (313) between January and March 2024?",  response should be similar to this but in dictionary format:

    start_date: 01/01/2024
    end_date: 31/01/2024
    group_method: 'Post Code Level'
    all_post_code: False
    all_customers: None
    selected_postcodes: [NG (313)]
    selected_customers:  []

    Note : if someone mention last month or recent month,  keep it November 2024 

    Steps to Answer for LLM:

    Identify the Time Frame (start_date and end_date):
    Look for explicit dates or ranges in the question.
    If not provided, default to the dataset’s min and max shipment dates (df['SHIPPED_DATE'].min() and df['SHIPPED_DATE'].max()).
    Determine the Grouping Method (group_method):
    If the question focuses on a specific user or group of users, set the group_method to 'Customer Level'.
    If it focuses on zip codes or regions, set the group_method to 'Post Code Level'.
    Check for Mentions of Post Codes or Customers:
    Extract any specific users (customers) or post codes mentioned in the question.
    If none are mentioned:
    For Customer Level, set all_customers = True and leave selected_customers empty.
    For Post Code Level, set all_post_code = True and leave selected_postcodes empty.
    Construct the Output:
    Return the appropriate values for:
    start_date
    end_date
    group_method
    all_post_code
    all_customers
    selected_postcodes
    selected_customers

    rerturn these variables in dictionary format keeping all these variables as keys.
    """
    )

    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4")

    # Create the Retrieval QA chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


    from langchain.agents import initialize_agent, Tool, AgentType

    # Combine tools and retrieval chain
    tools = [
        Tool(
            name="Document Retrieval",
            func=lambda q: retrieval_qa_chain({"query": q})["result"],
            description="Retrieve knowledge from the database."
        ),
        cost_consolidation_tool
    ]

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    def chatbot_agentic_rag():
        print("Agentic RAG Chatbot is running! Type 'exit' to quit.")
        user_query = query
        if user_query.lower() == "exit":
            print("Chatbot session ended.")
        response = agent.run(user_query)
        print(f"Bot: {response}")

    return chatbot_agentic_rag()