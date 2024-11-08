import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import time
from io import StringIO, BytesIO
import requests
from PIL import Image
from docx import Document
from docx.shared import Inches
import base64
# pip install python-docx

# Streamlit app configuration
st.set_page_config(
    page_title="Perrigo GenAI Answer Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo = Image.open("perrigo-logo.png")
st.image(logo, width=120)

# Custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_data_source' not in st.session_state:
    st.session_state.current_data_source = None

def reset_app_state():
    """Reset the app state when data source changes"""
    st.session_state.initialized = False
    if 'df' in st.session_state:
        del st.session_state.df
        
def save_figure_to_image(fig):
    """Convert matplotlib figure to bytes for Word document."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def create_word_document(chat_history):
    """Create a Word document from the analysis history."""
    doc = Document()
    doc.add_heading('Data Analysis History Report', 0)
    
    # Add generation timestamp
    doc.add_paragraph(f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    doc.add_paragraph('-' * 50)
    
    # Add each analysis to the document
    for idx, chat in enumerate(reversed(chat_history), 1):
        # Add query section
        doc.add_heading(f'Query: {chat["query"]}', level=1)
        doc.add_paragraph(f'Time: {chat["timestamp"]}')
        
        # Add approach section if exists
        if chat['approach']:
            doc.add_heading('Approach:', level=2)
            doc.add_paragraph(chat['approach'])
        
        # Add results section if exists
        if chat['answer']:
            doc.add_heading('Results:', level=2)
            doc.add_paragraph(chat['answer'])
        
        # Add visualization if exists
        if chat['figure']:
            doc.add_heading('Visualization:', level=2)
            image_stream = save_figure_to_image(chat['figure'])
            doc.add_picture(image_stream, width=Inches(6))
        
        # Add separator between analyses
        doc.add_paragraph('-' * 50)
    
    return doc

def download_word_doc():
    """Create and return a download link for the Word document."""
    if not st.session_state.chat_history:
        st.warning("No analysis history to export!")
        return
    
    # Create Word document
    doc = create_word_document(st.session_state.chat_history)
    
    # Save document to bytes
    doc_bytes = BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    
    # Create download button
    st.download_button(
        label="📥 Download Analysis History",
        data=doc_bytes.getvalue(),
        file_name=f"analysis_history_{time.strftime('%Y%m%d_%H%M%S')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

def extract_code_segments(response_text):
    """Extract code segments from the API response using regex."""
    segments = {}
    
    # Extract approach section
    approach_match = re.search(r'<approach>(.*?)</approach>', response_text, re.DOTALL)
    if approach_match:
        segments['approach'] = approach_match.group(1).strip()
    
    # Extract content between <code> tags
    code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
    if code_match:
        segments['analysis_code'] = code_match.group(1).strip()
    
    # Extract content between <chart> tags
    chart_match = re.search(r'<chart>(.*?)</chart>', response_text, re.DOTALL)
    if chart_match:
        segments['chart_code'] = chart_match.group(1).strip()
    
    # Extract content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if answer_match:
        segments['answer_template'] = answer_match.group(1).strip()
    
    return segments

def execute_analysis(df, response_text):
    """Execute the extracted code segments on the provided dataframe and store formatted answer."""
    results = {
        'approach': None,
        'answer': None,
        'figure': None
    }
    
    try:
        # Extract code segments
        segments = extract_code_segments(response_text)
        
        if not segments:
            st.error("No code segments found in the response")
            return results
        
        # Store the approach
        if 'approach' in segments:
            results['approach'] = segments['approach']
        
        # Create a single namespace for all executions
        namespace = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns}
        
        # Execute analysis code and answer template
        if 'analysis_code' in segments and 'answer_template' in segments:
            combined_code = f"""
{segments['analysis_code']}

# Format the answer template
answer_text = f'''{segments['answer_template']}'''
"""
            exec(combined_code, namespace)
            results['answer'] = namespace.get('answer_text')
        
        # Execute chart code if present
        if 'chart_code' in segments:
            plt.figure(figsize=(10, 6))
            exec(segments['chart_code'], namespace)
            fig = plt.gcf()
            results['figure'] = fig
            plt.close()
        
        return results
        
    except Exception as e:
        st.error(f"Error during execution: {e}")
        return results

def analyze_data_with_execution(df, question, api_key):
   
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        
You are an AI assistant tasked with analyzing a dataset to provide code for calculating the final answer and generating relevant visualization.
I will provide you with the data in dataframe format, as well as a question to answer based on the data.

Here is an example of what one row of the data looks like in json format but I will provide you with first 5 rows of the dataframe inside <data> tags:
{{
      "PROD_TYPE": "AMBCONTROL",
      "Customer": "GR & MM BLACKLEDGE PLC", 
      "SHIPPED_DATE": "01-01-2024",
      "Total_Orders": 2,
      "Total_Pallets": 2,
      "Distance": 134.5,
      "Cost": 102.8,
      "SHORT_POSTCODE": "PR"}}

<data>
{{df.head().to_string()}}
</data>

Some key things to note about the data:
- The "PROD_TYPE" column includes 2 values, either "AMBIENT" or "AMBCONTROL"
- The "SHIPPED_DATE" column ranges from Jan 2024 to Aug 2024 and is in dd-mm-yyyy format
- The "Cost" is in Pounds(£)

Here is the question I would like you to answer using this data:
<question>
{question}
</question>

To answer this, first think through your approach inside <approach> tags. Break down the steps you
will need to take and consider which columns of the data will be most relevant. Here is an example:
<approach>
To answer this question, I will need to:
1. Calculate the total number of orders and pallets across all rows
2. Determine the average distance and cost per order
3. Identify the most common PROD_TYPE and SHORT_POSTCODE
</approach>

Then, write the Python code needed to analyze the data and calculate the final answer inside <code> tags. Assume input dataframe as 'df'
Be sure to include any necessary data manipulation, aggregations, filtering, etc. Return only the Python code without any explanation or markdown formatting.
For decimal answers round them to 1 decimal place.

Generate Python code using matplotlib and/or seaborn to create an appropriate chart to visualize the relevant data and support your answer.
For example if user is asking for postcode with highest cost then a relevant chart can be a bar chart showing top 10 postcodes with highest total cost arranged in decreasing order.
Specify the chart code inside <chart> tags.
When working with dates:

Always convert dates to datetime using pd.to_datetime() with explicit format
For grouping by month, use dt.strftime('%Y-%m') instead of dt.to_period()
Sort date-based results chronologically before plotting

The visualization code should follow these guidelines:

Start with these required imports:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Use standard chart setup:
# Set figure size and style
plt.figure(figsize=(8, 5))
# Set seaborn default style and color palette
sns.set_theme(style="whitegrid")  
sns.set_palette('pastel')

For time-based charts:


Use string dates on x-axis (converted using strftime)
Rotate labels: plt.xticks(rotation=45, ha='right')
Add gridlines: plt.grid(True, alpha=0.3)

For large numbers:
Format y-axis with K/M suffixes using:

Always include:

Clear title (plt.title())
Axis labels (plt.xlabel(), plt.ylabel())
plt.tight_layout() at the end


For specific chart types:

Time series: sns.lineplot() with marker='o'
Rankings: sns.barplot() with descending sort
Comparisons: sns.barplot() or sns.boxplot()
Distributions: sns.histplot() or sns.kdeplot()

Return only the Python code without any explanation or markdown formatting.

Finally, provide the answer to the question in natural language inside <answer> tags. Be sure to
include any key variables that you calculated in the code inside {{}}.
                    """
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0
    }

    try:
        with st.spinner("Analyzing data..."):
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                  headers=headers, 
                                  json=payload)
            
            if response.status_code != 200:
                st.error(f"Error: Received status code {response.status_code}")
                st.error(f"Response content: {response.text}")
                return None
            
            response_json = response.json()
            response_content = response_json['choices'][0]['message']['content']
            
            # Execute the code segments and get results
            results = execute_analysis(df, response_content)
            
            # Store in chat history
            st.session_state.chat_history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": question,
                "approach": results['approach'],
                "answer": results['answer'],
                "figure": results['figure'],
                "raw_response": response_content
            })
            
            return results
            
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return None

def load_default_data():
    """Load the default CSV file."""
    try:
        return pd.read_csv('Data.csv', parse_dates=['SHIPPED_DATE'], dayfirst=True)
    except Exception as e:
        st.error(f"Error loading default data: {str(e)}")
        return None

def display_analysis_results(results):
    """Display the analysis results in a structured format."""
    if results['approach']:
        st.subheader("Approach")
        st.write(results['approach'])
    
    if results['answer']:
        st.subheader("Analysis Results")
        st.write(results['answer'])
    
    if results['figure']:
        st.pyplot(results['figure'])

def main():
    st.title("GenAI Answer Bot")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        st.subheader("1. API Key")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        
        # Data source selection
        st.subheader("2. Data Source")
        data_source = st.radio(
            "Choose Data Source:",
            ["Use Default Data", "Upload Custom File"]
        )
        
        # Reset state if data source changes
        if st.session_state.current_data_source != data_source:
            st.session_state.current_data_source = data_source
            reset_app_state()
        
        df = None
        if data_source == "Use Default Data":
            df = load_default_data()
            if df is not None:
                st.success("Default data loaded successfully!")
                st.session_state.df = df
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, parse_dates=['SHIPPED_DATE'], dayfirst=True)
                    st.success("Custom file loaded successfully!")
                    st.session_state.df = df
                except Exception as e:
                    st.error(f"Error loading custom file: {str(e)}")
    
    # Main content area
    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to get started.")
        return
    
    if 'df' not in st.session_state:
        if data_source == "Use Default Data":
            st.error("Default data file not found. Please check if 'Data.csv' exists.")
        else:
            st.info("Please upload your CSV file in the sidebar.")
        return
    
    # Display sample data
    with st.expander("📊 View Sample Data"):
        display_df = st.session_state.df.copy()
        display_df['SHIPPED_DATE'] = display_df['SHIPPED_DATE'].dt.strftime('%Y-%m-%d')
        display_df = display_df.set_index('PROD_TYPE')
        st.dataframe(display_df.head(), use_container_width=True)
    
    # Query interface
    st.subheader("💬 Ask Questions About Your Data")
    
    # Sample queries
    sample_queries = [
        "Which postcode results in the highest total cost?",
        "What is the monthly trend in total cost?",
        "What is the average cost per pallet for each PROD TYPE and how does it vary across the following SHORT_POSTCODE regions: CV, NG, NN, RG?",
        "Identify the distribution of cost per pallet, is it normally distributed?",
        "Generate a radar chart of average pallets per order for the top 15 postcodes with maximum average cost per order.",
        "Generate the boxplot distribution for pallets of the top 8 customers by total orders.",
        "For ambient product type, which are the top 5 customers with total orders > 10 and highest standard deviation in cost per pallet?",
        "What is the trend in cost over time and plot forecasted cost using 3-month exponential smoothing?",
        "Perform a hypothesis test to analyze if average cost per order differs significantly with product type.",
        "Create a regression line for cost per order and distance along with R squared.",
        "What is the distribution of cost in percentiles?",
        "How does the cost per order vary with distance within each PROD TYPE?",
        "Find the top 5 customers by total pallets shipped and compare their average cost per pallet and distance traveled.",
        "Identify the SHORT_POSTCODE areas with the highest total shipping costs and also mention their cost per pallet.",
        "Which customer has the highest total shipping cost over time, and how does its cost trend vary by month?",
        "What is the order frequency per week for the last 2 months?",
        "What is the total cost for ambient product type in January 2024?",
        "How has the cost per pallet evolved over the last 3 months?",
        "What is the average cost per pallet for each product type?"
    ]


    
    selected_query = st.selectbox(
        "Select a sample query or write your own below:",
        [""] + sample_queries,
        key="query_select"
    )
    
    query = st.text_area(
        "Enter your query:",
        value=selected_query,
        height=100,
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔍 Analyze")
    
    if submit_button and query:
        start_time = time.time()
        
        results = analyze_data_with_execution(st.session_state.df, query, api_key)
        
        if results:
            display_analysis_results(results)
            
            end_time = time.time()
            time_taken = end_time - start_time
            st.info(f"Analysis completed in {time_taken:.1f} seconds")
    
    
    # Display analysis history with download button
    if st.session_state.chat_history:
        col1, col2 = st.columns([6, 2])
        with col1:
            st.subheader("📜 Analysis History")
        with col2:
            download_word_doc()
            
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(
                f"Query {len(st.session_state.chat_history) - idx}: {chat['query'][:50]}...",
                expanded=False
            ):
                st.markdown(f"**🕒 Time:** {chat['timestamp']}")
                st.markdown("**🔍 Query:**")
                st.write(chat['query'])
                
                if chat['approach']:
                    st.markdown("**🎯 Approach:**")
                    st.write(chat['approach'])
                
                if chat['answer']:
                    st.markdown("**💡 Results:**")
                    st.write(chat['answer'])
                
                if chat['figure']:
                    st.pyplot(chat['figure'])

if __name__ == "__main__":
    main()