import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_KEY')},
    temperature=0.7,
)

class Message(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def process_node(state:Message)->Message:
    # Invoke messages
    response = llm.invoke(state["messages"]) 
    # Append to state["messages"] the AI response
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(Message)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Your message: ")
while user_input != "exit":
    # Save human inputs
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    print(result["messages"])
    conversation_history = result["messages"]

    user_input = input("Your message: ")


