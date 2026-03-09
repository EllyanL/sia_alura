from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

model = ChatOpenAI(
    base_url=f"http://{os.getenv('IP_ADDRESS')}/v1",
    api_key="EMPTY",
    model="openai/gpt-oss-20b",  
    temperature=0.7,
).bind(response_format={"type": "json_object"})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente de League of Legends que responde sempre em formato JSON. Responda apenas o que foi perguntado, de forma direta e sem explicações extras."),
        ("placeholder", "{History}"),
        ("human", "{query}"),
    ]
)

chain = prompt | model | StrOutputParser()


memoria = {}
sessao = "teste"

def get_historico(sessao_id):
    if sessao_id not in memoria:
        memoria[sessao_id] = InMemoryChatMessageHistory()
    return memoria[sessao_id]

lista_perguntas = [
    "Quando eu estiver de MAGO  , e não tiver muito tank, devo buildar dano ou penetração?",
    "Me de um exemplo de item? "
]


chain_com_historico = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_historico,
    input_messages_key="query",
    history_messages_key="History",
)

for uma_pergunta in lista_perguntas:
    response = chain_com_historico.invoke(
        {"query": uma_pergunta},
        config={"configurable": {"session_id": sessao}}
    )
    print("Usuário: ", uma_pergunta)
    print("IA: ", response, "\n")