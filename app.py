import os, textwrap
import gradio as gr
from google import genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-flash-latest"             

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an AI Employee Copilot for office workers.
You help with repetitive administrative tasks such as:
- drafting emails (leave requests, status updates, client communication)
- summarizing meeting notes or documents
- generating weekly status reports or simple documentation.
Guidelines:
- Be concise and professional.
- If there are uploaded documents, use their content as context.
- If the user request is unclear, ask one clarifying question instead of guessing.
- Respond in plain text (no markdown tables).
- At the end of your answer, add a line starting with 'Explanation:' that
  briefly explains what you did.
"""

# ---------- Helpers ----------

def normalize_files(files):
    if files is None:
        return []
    if isinstance(files, list):
        return files
    return [files]

def build_context_from_files(files):
    files_list = normalize_files(files)
    summaries = []
    for f in files_list:
        try:
            with open(f.name, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
            summary = textwrap.shorten(content, width=1500, placeholder="...")
            summaries.append(f"File: {os.path.basename(f.name)}\n{summary}")
        except Exception as e:
            summaries.append(f"File: {os.path.basename(f.name)}\n[Could not read file: {e}]")
    return "\n\n".join(summaries)

# ---------- Main copilot logic using GenAI SDK ----------

def employee_copilot_genai(task_description, files):
    if not task_description or task_description.strip() == "":
        return (
            "Please describe your task (e.g. 'Draft a leave email', "
            "'Summarize meeting notes', 'Create weekly status report').",
            "Asked user to provide a non-empty task description.",
        )

    context = build_context_from_files(files)

    user_prompt = f"""
Employee task description:
{task_description}
Context from uploaded files (if any):
{context if context else "[No files provided]"}
Your goals:
- Understand what the employee wants.
- Choose a suitable action (email draft, summary, report, etc.).
- Produce a professional, ready-to-use response.
- End with a line starting with 'Explanation:' that briefly describes what you did.
"""

    fallback_main = (
        "Something went wrong while generating the response. "
        "Please check your API key, model name, or internet connection and try again."
    )
    fallback_explanation = "Model call failed; returned a fallback error message."

    try:
        # Simple usage: pass strings to contents, SDK wraps them internally [web:83][web:92].
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SYSTEM_PROMPT, user_prompt],
        )
        text = response.text.strip()
    except Exception as e:
        error_msg = f"Error calling Gemini via GenAI SDK: {e}"
        return error_msg, fallback_explanation

    main_output = text
    explanation = fallback_explanation

    if "Explanation:" in text:
        parts = text.split("Explanation:", 1)
        main_output = parts[0].strip()
        explanation = parts[1].strip()

    if not explanation:
        explanation = "Generated a response based on your task description and any uploaded context."

    return main_output, explanation

# ---------- Gradio UI ----------

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Employee Copilot")
    gr.Markdown(
        "Describe your administrative task (e.g. 'Draft a leave email', "
        "'Summarize today’s meeting', 'Create weekly status report') "
        "and optionally upload related files (policies, notes, etc.)."
    )

    with gr.Row():
        task_input = gr.Textbox(
            label="Describe your task",
            placeholder="Example: Draft a leave email to my manager for 2 days sick leave next week",
            lines=3,
        )

    files_input = gr.File(
        label="Upload relevant files (optional, text files work best)",
        file_count="multiple",
        type="filepath",
    )

    generate_btn = gr.Button("Run Copilot")

    output_text = gr.Textbox(
        label="Generated output",
        lines=18,
    )

    explanation_text = gr.Textbox(
        label="What the copilot did",
        lines=3,
    )

    generate_btn.click(
        fn=employee_copilot_genai,
        inputs=[task_input, files_input],
        outputs=[output_text, explanation_text],
    )

demo.launch(share=True, debug=True)
