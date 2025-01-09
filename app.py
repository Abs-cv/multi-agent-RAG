import os
from getpass import getpass
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict, Literal
import functools
import operator
from typing_extensions import TypedDict

# Utility function to safely set environment variables
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}: ")

# Set API keys securely (remove hardcoded keys)
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Enable LangSmith tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"

# Create agents using helper functions
def create_agent(llm, tools, system_message: str):
    """Create an agent with specified tools and system message."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful AI assistant, collaborating with other assistants.
            Use the provided tools to progress towards answering the question.
            If you are unable to fully answer, another assistant will help where you left off.
            Prefix your response with FINAL ANSWER if you have the final deliverable.
            You have access to the following tools: {tool_names}.
            {system_message}
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# Define tools
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl(code: Annotated[str, "The Python code to execute to generate your chart."]):
    """Execute Python code securely."""
    try:
        result = repl.run(code)
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

# Define agent state class
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Define helper function to create agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

# Initialize LLM and create agents
llm = ChatOpenAI(model="gpt-4-1106-preview")

research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible to the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

# Define tool node
tools = [tavily_tool, python_repl]
tool_node = ToolNode(tools)

# Define router logic
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"

# Define the state graph
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"Researcher": "Researcher", "chart_generator": "chart_generator"},
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()

# Sample invocation with multiple examples
examples = [
    "Fetch the UK's GDP over the past 5 years, then draw a line graph of it.",
    "Provide a summary of climate change statistics for the last decade and visualize the data.",
    "Analyze the trends in global renewable energy adoption rates over the past 5 years and create a bar chart."
]

for example in examples:
    print(f"Example Task: {example}")
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=example
                )
            ],
        },
        {"recursion_limit": 150},
    )
    for s in events:
        print(s)
        print("----")
