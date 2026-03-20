from crewai import Agent, LLM, Crew, Task
import os
from variable import keywords, sample_contract, concerto_model
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="gemini/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

template_agent = Agent(
    role="Template Agent",
    goal="Generate a contract template by mapping the extracted variables to their corresponding placeholders",
    backstory="You are an expert in Accord Project template syntax who builds contracts purely from variable definitions",
    llm=llm,
    verbose=False
)
template_task = Task(
    description=f"""Using the concerto model fields {concerto_model} as the ONLY valid placeholder names,
    and the contract {sample_contract} as the structure,
    generate a valid Accord Project template.
    ONLY use field names defined in the Concerto model as placeholders.
    Do NOT use raw keywords as placeholders.""",
    expected_output="A clean contract template using only Concerto model field names as [{placeholder}] variables",
    agent=template_agent
)

crew = Crew(agents=[template_agent], tasks=[template_task], llm=llm)
result = crew.kickoff()
print(result)