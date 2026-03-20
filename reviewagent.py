from crewai import Agent, LLM, Crew, Task
import os, re, json, time, subprocess
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="gemini/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
concerto_model_output = """
namespace employment@1.0.0

concept EmploymentContract {
  o String companyName
  o String contractorName
  o Double compensation
  o Integer durationMonths
  o Integer noticePeriodDays
}
"""

template_output = """
This Agreement is made between {{companyName}} and {{contractorName}}.
The contract duration is {{durationMonths}} months.
Compensation is {{compensation}} payable quarterly.
Either party may terminate with {{noticePeriodDays}} days written notice.
"""

def extract_model_fields(model: str) -> dict:
    """Returns {field_name: type} from a concerto model string."""
    return {
        m.group(2): m.group(1)
        for m in re.finditer(r'o\s+(\w+)\s+(\w+)', model)
    }

def extract_template_vars(template: str) -> set:
    """Returns all {{placeholder}} names from a TemplateMark template."""
    return set(re.findall(r'\{\{(\w+)\}\}', template))

def validate_field_types(model_fields: dict) -> list:
    """
    Checks that field types are appropriate Concerto primitives.
    Returns a list of warnings for any unexpected types.
    """
    valid_types = {"String", "Double", "Integer", "Long", "Boolean", "DateTime"}
    warnings = []
    for field, ftype in model_fields.items():
        if ftype not in valid_types:
            warnings.append({
                "field": field,
                "message": f"'{ftype}' is not a standard Concerto primitive type"
            })
    return warnings

def run_concerto_cli(model: str) -> tuple[bool, str]:
    """
    Runs concerto-cli validation on the model string.
    Returns (is_valid, error_message).
    Falls back gracefully if concerto-cli is not installed.
    """
    try:
        with open("/tmp/model.cto", "w") as f:
            f.write(model)
        proc = subprocess.run(
            ["concerto", "validate", "--model", "/tmp/model.cto"],
            capture_output=True, text=True, timeout=15
        )
        return proc.returncode == 0, proc.stderr.strip()
    except FileNotFoundError:
        return None, "concerto-cli not installed — skipping CLI validation"
    except subprocess.TimeoutExpired:
        return None, "concerto-cli timed out"

start_time = time.time()
model_fields      = extract_model_fields(concerto_model_output)
template_vars     = extract_template_vars(template_output)
missing_in_model  = template_vars - model_fields.keys()  
unused_in_template = model_fields.keys() - template_vars  
type_warnings     = validate_field_types(model_fields)
cli_valid, cli_error = run_concerto_cli(concerto_model_output)
total_vars  = len(template_vars)
matched     = len(template_vars & model_fields.keys())
accuracy    = round((matched / total_vars) * 100, 2) if total_vars > 0 else 0.0

deterministic_status = "PASS" if not missing_in_model else "TEMPLATE_VAR"

checklist = {
    "model_fields":          model_fields,
    "template_vars":         list(template_vars),
    "missing_in_model":      list(missing_in_model),
    "unused_fields":         list(unused_in_template),
    "type_warnings":         type_warnings,
    "cli_valid":             cli_valid,
    "cli_error":             cli_error,
    "deterministic_status":  deterministic_status,
    "accuracy_score":        accuracy,
    "matched":               matched,
    "total_vars":            total_vars,
}


review_agent = Agent(
    role="Review Agent",
    goal="Classify errors and write clear human-readable explanations based on pre-computed validation results",
    backstory=(
        "You are a senior Accord Project reviewer. You receive pre-computed "
        "validation results and your job is to classify errors, write clear "
        "human-readable explanations, and decide which agent should retry. "
        "You never re-derive results — the Python layer is ground truth."
    ),
    llm=llm,
    verbose=False
)

review_task = Task(
    description=f"""
    The following validation has already been run deterministically in Python.
    Do NOT re-derive these results — treat them as ground truth.

    Checklist:
    {json.dumps(checklist, indent=2)}

    Your job:
    1. Accept deterministic_status as the final status — do not override it
    2. For each entry in missing_in_model: classify as TEMPLATE_VAR and write
       a one-line explanation of what the TemplateAgent needs to fix
    3. For each entry in type_warnings: include as a MODEL_TYPE warning
    4. Note unused_fields as informational warnings (not errors)
    5. If cli_valid is False: include the cli_error as a CLI_ERROR entry
    6. If cli_valid is None: note that CLI validation was skipped
    7. Set retry_agent:
       - "TemplateAgent" if status is TEMPLATE_VAR
       - "ModelAgent"    if status is MODEL_MISMATCH
       - null            if status is PASS

    Respond ONLY with valid JSON — no markdown, no preamble, no explanation outside the JSON:
    {{
      "status": "PASS" | "MODEL_MISMATCH" | "TEMPLATE_VAR",
      "accuracy_score": <number from checklist>,
      "matched": <number from checklist>,
      "total_vars": <number from checklist>,
      "errors": [
        {{"field": "fieldName", "type": "TEMPLATE_VAR" | "MODEL_TYPE" | "CLI_ERROR", "message": "..."}}
      ],
      "warnings": [
        {{"field": "fieldName", "message": "..."}}
      ],
      "retry_agent": "ModelAgent" | "TemplateAgent" | null,
      "summary": "One sentence summary of the review result"
    }}
    """,
    expected_output="Valid JSON with status, accuracy_score, errors, warnings, retry_agent, and summary",
    agent=review_agent
)

crew   = Crew(agents=[review_agent], tasks=[review_task])
result = crew.kickoff()

elapsed = round(time.time() - start_time, 2)


try:
    raw = str(result).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    report = json.loads(raw)
    print("\nReview Agent Report")
    print(f"  status         : {report['status']}")
    print(f"  accuracy score : {report['accuracy_score']}%  "
          f"({report['matched']}/{report['total_vars']} placeholders matched)")
    print(f"  latency        : {elapsed}s")
    print(f"  retry_agent    : {report['retry_agent']}")
    print(f"  summary        : {report['summary']}")

    if report['errors']:
        print("\n  errors:")
        for e in report['errors']:
            print(f"    [{e['type']}] {e['field']} — {e['message']}")
    else:
        print("\n  errors        : none")

    if report.get('warnings'):
        print("\n  warnings:")
        for w in report['warnings']:
            print(f"    [WARN] {w['field']} — {w['message']}")
    else:
        print("  warnings      : none")

    if cli_valid is None:
        print(f"\n  cli validation : skipped — {cli_error}")
    elif cli_valid:
        print("\n  cli validation : PASS")
    else:
        print(f"\n  cli validation : FAIL — {cli_error}")

except json.JSONDecodeError:
    print("\n[ERROR] Review agent returned non-JSON output:")
    print(result)