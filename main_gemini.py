"""
Simple CrewAI project with Gemini: Topic Explainer
--------------------------------------------------
One Gemini LLM is shared by two agents:
  1. Researcher -> collects key points about a topic
  2. Writer     -> turns them into a short, beginner-friendly explanation

Setup:
    pip install crewai
    Get a free key from https://aistudio.google.com/apikey

    Linux/Mac:  export GEMINI_API_KEY="your-key"
    Windows:    set GEMINI_API_KEY=your-key
    Colab:      import os; os.environ["GEMINI_API_KEY"] = "your-key"

Run:
    python main_gemini.py "Edge AI in EV charging"
"""

import os
import sys
from crewai import Agent, Task, Crew, Process, LLM

# Option 1: read the key from the environment (recommended)
# Option 2: paste it here for a quick test (don't share or commit this file then)
API_KEY = os.getenv("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_API_KEY_HERE")

# ---------- 1. ONE LLM (Gemini), shared by every agent ----------
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0.3,
)

# ---------- 2. Agents ----------
researcher = Agent(
    role="Research Analyst",
    goal="Find the most important facts about {topic}",
    backstory="You are a careful analyst who breaks any topic into clear, accurate key points.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Explain {topic} simply for a college student",
    backstory="You turn dense research notes into short, clear, engaging explanations.",
    llm=llm,
    verbose=True,
)

# ---------- 3. Tasks ----------
research_task = Task(
    description="List 5 key points about {topic}: what it is, why it matters, how it works, "
                "one real-world use case, and one limitation.",
    expected_output="A bullet list of 5 concise key points.",
    agent=researcher,
)

writing_task = Task(
    description="Using the research notes, write a 150-word explanation of {topic} "
                "with a short title and a one-line summary at the end.",
    expected_output="A titled, ~150-word explanation ending with a one-line summary.",
    agent=writer,
    context=[research_task],
)

# ---------- 4. Crew ----------
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Edge AI"
    result = crew.kickoff(inputs={"topic": topic})
    print("\n===== FINAL OUTPUT =====\n")
    print(result)
