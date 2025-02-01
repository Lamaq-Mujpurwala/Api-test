from langchain_community.tools import YouTubeSearchTool
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor


llm = ChatGroq(
    api_key="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",
    model="deepseek-r1-distill-llama-70b",
    temperature=0.7, #initially was 0.7
    verbose=True,
)

#initially None
yt_tool = YouTubeSearchTool()

def call_yt_tool(concept: str):
  
    return yt_tool.run(tool_input=f"{concept} , 3")

youtube_tool = Tool(
    name="YouTubeSearch",
    func=lambda concept: call_yt_tool(concept),
    description=(
        "Use this tool to search YouTube for learning resources. "
        "Provide a concept as input, and it will return up to 3 videos explaining the concept."
    ),
)
  
new_tools = [youtube_tool]


new_prompt_template = """
You're a data science expert assistant. Teach {concept} with clear explanations and real YouTube links from YouTubeSearch.

**Instructions:**
**Main Concept**:
   - 1-sentence definition
   - 1-line importance
   - 3 learning steps

**Key Sub-Concepts** (if needed):
   - List 2-3 prerequisites
   - Brief explanation each

**Videos**:
   - Use YouTubeSearch once per concept
   - Provide 3 relevant videos with links
   - Return ACTUAL YouTube links (3 per concept)
   - Include title + URL + 1-line benefit

**Format Requirements:**
- Always show full YouTube URLs
- No placeholder links (e.g., ...)
- Use bullet points
- Include analogies

Example for "XGBoost":
1. **Main Concept**: Ensemble method using boosted decision trees
   - Why: Dominates ML competitions for structured data
   - Learn: 
     1. Basic decision trees
     2. Gradient boosting theory
     3. XGBoost optimizations

   **Videos**:
   • "XGBoost Explained in Python" (https://youtu.be/123XYZ) - Practical implementation
   • "Math Behind XGBoost" (https://youtu.be/456ABC) - Theoretical foundation
   • "XGBoost vs LightGBM" (https://youtu.be/789DEF) - Performance comparison

2. **Sub-Concepts**:
   - Decision Trees: 
     • "Decision Tree Basics" (https://youtu.be/101GHJ) - Splitting criteria visualization
     • "Tree Pruning" (https://youtu.be/202KLK) - Prevent overfitting

{agent_scratchpad}
"""


new_concept_prompt = PromptTemplate(
    template=new_prompt_template,
    input_variables=["concept","agent_scratchpad"]
)


new_concept_agent = create_openai_tools_agent(llm , new_tools , new_concept_prompt)


new_concept_agent_executor = AgentExecutor(agent=new_concept_agent , tools=new_tools , verbose=True)
