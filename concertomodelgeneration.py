from crewai import Agent, LLM, Crew, Task
import os

llm = LLM(model="gemini/gemini-2.5-flash", api_key=os.environ.get("GOOGLE_API_KEY"))
concerto_model = Agent(
    role="Write a concerto model using the keywords given",
    goal="get a sensible concerto code here upon using the keywords",
    verbose=True,
    memory=False,
    llm=llm,
    backstory=(
        "You are an expert in Accord Project's Concerto modeling language with deep knowledge "
        "of contract law and data modeling. You specialize in translating legal contract keywords "
        "— such as parties, obligations, and asset types — into clean, valid Concerto model code "
        "that follows standard namespace and class structure conventions."
    ),
)

keywords = [
    "Agreement",
    "March 1 2026",
    "ABC Corp",
    "Company",
    "John Doe",
    "Contractor",
    "software development services",
    "Contractor provide services",
    "12 months",
    "Compensation 100000",
    "payable quarterly",
    "Contractor maintain confidentiality",
    "Company data",
    "intellectual property",
    "party terminate Agreement",
    "30 days written notice",
    "Liability limited",
    "direct damages",
]

task = Task(
    description=f"Generate a Concerto model using these keywords: {keywords}",
    expected_output="A valid Concerto model file with namespaces, classes and fields",
    agent=concerto_model,
)

crew = Crew(agents=[concerto_model], tasks=[task])
result = crew.kickoff()
print(result.raw)
