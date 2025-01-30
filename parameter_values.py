import openai
import re
import ast
import pandas as pd

def get_chatgpt_response(api_key, instructions, user_query):
    """
    Sends a query to OpenAI's ChatCompletion API with the given instructions and user query.
    """
    # Set the API key
    openai.api_key = api_key
 
    try:
        # Send the query to OpenAI ChatCompletion API
        response = openai.chat.completions.create(
            model="gpt-4o",  # Specify the GPT-4 model
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_query}
            ],
            max_tokens=500,  # Adjust token limit based on your needs
            temperature=0.7  # Adjust for creativity (0.7 is a balanced value)
        )
        # Extract and return the assistant's response
        return response.choices[0].message.content

    except openai.OpenAIError as e:
        # Handle OpenAI-specific errors
        return f"An error occurred with the OpenAI API: {str(e)}"

    except Exception as e:
        # Handle other exceptions (e.g., network issues)
        return f"An unexpected error occurred: {str(e)}"


def ask_openai(selected_customers,selected_postcodes,customers,postcodes):
    """
    Sends a question and data context to OpenAI API for processing.
    """
    # Formulate the prompt
    prompt = f"""You are given four lists as inputs:

    {selected_customers}: A list of selected customer names.
    {selected_postcodes}: A list of selected postcodes.
    {customers}: A list of unique customer names.
    {postcodes}: A list of unique postcodes corresponding to the customers.
    
    Your task is to find the best match for each item in {selected_customers} and {selected_postcodes} from the {customers} and {postcodes} lists respectively. The matching should be case-insensitive and prioritize similarity. If there are multiple possible matches, return the most suitable one.

    The output should consist of two separate lists:

    A list of matched customers.
    A list of matched postcodes.

    Example Input:
    selected_customers = ['Alloga', 'FORum', 'usa']  
    selected_postcodes = ['ng', 'Lu']  
    customers = ['ALLOGA UK', 'FORUM', 'USA', 'ALLOGA FRANCE', 'BETA PHARMA']  
    postcodes = ['NG', 'LU', 'NN', 'NZ', 'AK']

    Expected Output format:
    
    matched_customers: ['ALLOGA UK','ALLOGA FRANCE', 'FORUM', 'USA']
    matched_postcodes: ['NG', 'LU']
    


    Process the inputs {selected_customers}, {selected_postcodes}, {customers}, and {postcodes} and return the final answer that should contain only two lists with no explanation.

    <answer>
    matched_customers: ['ALLOGA UK','ALLOGA FRANCE', 'FORUM', 'USA']
    matched_postcodes: ['NG', 'LU']
    </answer>
    """
        
    # Call OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are an assistant skilled at answering questions about searching something"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.7
    )
    return response.choices[0].message.content


def get_parameters_values(api_key, query):

    instructions = """You are an AI assistant tasked with analyzing questions and based on that give value of certain variables:
    list of variables are below:
    start_date:
    end_date:
    group_method:
    all_post_code: 
    all_customers:
    selected_postcodes: 
    selected_customers:

    I will provide you a question to answer, based on the question you need to provide variable values .

    Here are some sample questions I would like you to answer:
    1. How can I optimize the shipment costs for user ALLOGA UK.
    2. Can you optimize costs for shipments to zip code NG between January and March 2024?

    To answer this, first think through your approach
    To answer this question, 
    1. You will need to find the start and end date first if it is not mentioned then start date will be 1st january 2023 and end date will be 30th November 2024
    2. Determine the group_method, whether it 'Customer Level' or 'Post Code Level'
    3. Determine the list of post codes or list of users that are mentioned in the question, if there is no mention of post code or users , then make all_post_code = False  if group method is Post Code Level otherwise keep it None, and  all_customers = False if group method is Customer Level otherwise keep it None.
    4. if there is a mention of certain users or zip codes, make a list of that.

    return the value of all the required variables based on the questions in json format.

    for example for the first question "How can I optimize the shipment costs for user ALLOGA UK." the response should be similar to this but in dictionary format:
    
    expected output format:

    start_date: 2023-01-01
    end_date: 2024-11-30
    group_method: 'Customer Level'
    all_post_code: None
    all_customers: False
    selected_postcodes: []
    selected_customers:  [ALLOGA UK]


    for the 2nd question "Can you optimize costs for shipments to zip code NG (313) between January and March 2024?",  response should be similar to this but in dictionary format:
    
    expected output format:

   
    start_date: 2024-01-01
    end_date: 2024-01-31
    group_method: 'Post Code Level'
    all_post_code: False
    all_customers: None
    selected_postcodes: [NG]
    selected_customers:  []
   

    Note : if someone mention last month or recent month,  keep it November 2024, and date format should be: yyyy-mm-dd

    strict instructions: The final output should be only in this format (no extra text or steps should be included in the output):

    { "start_date": "2024-11-01",
    "end_date": "2024-11-30",
    "group_method": "Post Code Level",
    "all_post_code": True,
    "all_customers": None,
    "selected_postcodes": [],
    "selected_customers": [] }

    """
    response = get_chatgpt_response(api_key, instructions, query)
    print(response)
    if response:
        try:
            extracted_code= eval(response)
            input=pd.read_excel("Complete Input.xlsx")
            customers=input["NAME"].unique()
            postcodes=input["SHORT_POSTCODE"].unique()
            selected_customers= extracted_code['selected_customers']
            selected_postcodes= extracted_code['selected_postcodes']
            answer = ask_openai(selected_customers,selected_postcodes,customers,postcodes)
            
            # Extract matched_customers
            customers_match = re.search(r"matched_customers:\s*(\[.*\])", answer)
            matched_customers = ast.literal_eval(customers_match.group(1)) if customers_match else []

            # Extract matched_postcodes
            postcodes_match = re.search(r"matched_postcodes:\s*(\[.*\])", answer)
            matched_postcodes = ast.literal_eval(postcodes_match.group(1)) if postcodes_match else []

            print(matched_customers)
            print(matched_postcodes)
            extracted_code['selected_customers']= matched_customers
            extracted_code['selected_postcodes']= matched_postcodes

            return extracted_code
    
    ### return default parameters:
        except: 
            default_param={
            "start_date": "01/01/2024",
            "end_date": "31/03/2024",
            "group_method": "Post Code Level",
            "all_post_code": False,
            "all_customers": None,
            "selected_postcodes": ["NG"],
            "selected_customers": [] }

            return default_param
