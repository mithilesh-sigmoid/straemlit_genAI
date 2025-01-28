import openai
import re

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


def get_parameters_values(api_key, query):
    """
    Prepares instructions for the AI and queries it for parameter values based on the user's question.
    """
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
    If not provided, default to the datasets min and max shipment dates (df['SHIPPED_DATE'].min() and df['SHIPPED_DATE'].max()).
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
    return these variables in dictionary format keeping all these variables as keys.
    """
    response = get_chatgpt_response(api_key, instructions, query)

    code_block = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    try:
        if code_block: 
            extracted_code = code_block.group(1).strip('') 
            return eval(extracted_code)
    
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
