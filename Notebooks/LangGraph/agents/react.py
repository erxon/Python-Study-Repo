"""
Reasoning and Acting Agent
"""
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # Appending messages instead of overwriting

@tool
def add(a: int, b:int):
    """ Simply adds two numbers """
    return a + b

@tool
def subtract(a: int, b: int):
    """ Get difference between two numbers """

    return a - b

@tool
def multiplication(a: int, b: int):
    """ Get product of two numbers """

    return a * b

tools = [add, subtract, multiplication]

model = ChatOllama(
    model="gpt-oss:120b-cloud",
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_KEY')},
    temperature=0.7,
).bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability")
    response = model.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}

# Conditional node
def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

# Create and Add Tool Node
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

# Set entry to agent
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

# Helper function
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 109 and 304 and Subtract 4 into it and multiply it to 100")]}
print_stream(app.stream(inputs, stream_mode="values"))
