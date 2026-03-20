from google import genai
import os
from dotenv import load_dotenv
import nltk

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

sample_contract = """
This Agreement is made on March 1, 2026, between ABC Corp ("Company") and 
John Doe ("Contractor"). The Contractor shall provide software development 
services for a period of 12 months. Compensation shall be $100,000, payable 
quarterly. The Contractor must maintain confidentiality of all Company data 
and intellectual property. Either party may terminate this Agreement with 
30 days written notice. Liability is limited to direct damages only.
"""


def extract_keywords_google(contract_text):
    prompt = f"""
    Extract all important keywords, clauses, obligations, and requirements
    from this contract. Remove common words like is, am, was, were, the, a, an, of, for.
    Return only a single unified clean list of keywords or phrases, no duplicates.

    Contract: "{contract_text}"
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


all_keywords = extract_keywords_google(sample_contract)
print(all_keywords)
