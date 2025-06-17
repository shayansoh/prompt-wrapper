import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("LLAMA_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_data(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM CAREER SITE:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from a career site's job posting.
            Your job is to extract the data and return them in JSON format containing the following properties: 
            - `role`, 
            - `company name`,
            - 'industry` (e.g., Technology, Finance, Healthcare),
            - `level` (e.g Student, Junior, Senior, Executive), 
            - `experience required` (estimateed in years), 
            - `skills` (array of skills), 
            - `location`, 
            - `pay range (min)` (if available, use integer),
            - `pay range (max)` (if available, use integer),
            - `employment type` (e.g., Full-time, Part-time, Contract),
            - `short description`

            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("ERROR: Context is too big. Please provide a smaller context or check the input format.")
        return res if isinstance(res, list) else [res]

    def define_response(self, job):
        sample_prompt = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are a specialized recruiter working for a top recruiting consulting agency. You 
            are tasked with preparing a role summary for the provided role so that you can easily
            your team can have an easier time at sourcing candidates for the role.

            Present the results from the JSON in a tabular format where the JSON property names
            are the column names and the values are the rows. Also, present the JSON in a valid JSON format.
            
            For anything missing in the JSON, use "N/A" as the value.
            
            ### JSON (NO PREAMBLE):
            ### TABLE (NO PREAMBLE):

            """
        )
        chain_prompt = sample_prompt | self.llm
        res = chain_prompt.invoke({"job_description": str(job)})
        return res.content

if __name__ == "__main__":
    print(os.getenv("LLAMA_KEY"))