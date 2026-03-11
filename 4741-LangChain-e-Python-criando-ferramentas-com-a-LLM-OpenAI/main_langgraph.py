from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from typing import Literal, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
import asyncio
load_dotenv()

model = ChatOpenAI(
    base_url=f"http://{os.getenv('IP_ADDRESS')}/v1",
    api_key="EMPTY",
    model="openai/gpt-oss-20b",
    temperature=0.7,
).bind(response_format={"type": "json_object"})

prompt_consultor_viego = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como um pro player de League of Legends especializado em Viego. Responda apenas com um JSON no formato: {{'resposta': 'sua explicação aqui'}}"),
    ("human", "{query}"),
])

prompt_consultor_zoe = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como um pro player de League of Legends especializado em Zoe. Responda apenas com um JSON no formato: {{'resposta': 'sua explicação aqui'}}"),
    ("human", "{query}"),
])

chain_consultor_viego = prompt_consultor_viego | model | JsonOutputParser()

chain_consultor_zoe = prompt_consultor_zoe | model | JsonOutputParser()

class Rota(TypedDict):
    consultor: Literal["viego", "zoe"]
    query: str

prompt_router = ChatPromptTemplate.from_messages([
    ("system", """Você é um roteador que decide qual consultor deve responder a pergunta.
    
    Siga estritamente estas regras:
    1. Se a pergunta for sobre o campeão Viego, responda: {{"consultor": "viego"}}
    2. Se a pergunta for sobre a campeã Zoe, responda: {{"consultor": "zoe"}}
    
    Não adicione nenhuma outra explicação, responda apenas o JSON."""),
    ("human", "{query}"),
])


roteador = prompt_router | model | JsonOutputParser()

class State(TypedDict):
    query:str
    champion:str
    response:str
    

async def node_router(state: State, config=RunnableConfig):
    return {"champion": roteador.invoke({"query": state["query"]}, config)}

async def node_viego(state: State, config=RunnableConfig):
    return {"response": chain_consultor_viego.invoke({"query": state["query"]}, config)}

async def node_zoe(state: State, config=RunnableConfig):
    return {"response": chain_consultor_zoe.invoke({"query": state["query"]}, config)}

def chose_node(state:State)->Literal["Viego", "Zoe"]:
    return "Viego" if state["champion"]["consultor"] == "viego" else "Zoe"  

graph = StateGraph(State)
graph.add_node("rotear", node_router)
graph.add_node("Viego", node_viego)
graph.add_node("Zoe", node_zoe) 

graph.add_edge(START, "rotear")
graph.add_conditional_edges("rotear", chose_node)
graph.add_edge("Viego",END)
graph.add_edge("Zoe",END)

app = graph.compile() 

async def main():
    response = await app.ainvoke(
        {"query": "Me explique a passiva do Viego"}
    )
    print(response["response"])

asyncio.run(main())