from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.globals import set_debug
import os 


set_debug(False)
load_dotenv()

class champion(BaseModel):
    name: str = Field(description="Nome do campeão")
    advantage: str = Field(description="Vantagem do campeão")

class counter(BaseModel):
    name: str = Field(description="Nome do campeão")
    advantage: str = Field(description="Vantagem do campeão")


parser_champion=JsonOutputParser(pydantic_object=champion)
parser_counter=JsonOutputParser(pydantic_object=counter)

model_prompt_champion = PromptTemplate(
    input_variables=["champ"],
    template="""Responda sempre em Portugues Brasileiro. Qual a vantagem do {champ}, para uma partida solo/duo de league of legends.{format}""",
    partial_variables={"format": parser_champion.get_format_instructions()}
)

model_prompt_counter = PromptTemplate(
    template="""Responda sempre em Portugues Brasileiro. Sugira um campeão counter de {name}, para uma partida solo/duo de league of legends, apenas um campeão.{format}""",
    partial_variables={"format": parser_counter.get_format_instructions()}
)

model = ChatOpenAI(
    base_url=f"http://{os.getenv('IP_ADDRESS')}/v1",
    api_key="EMPTY",
    model="openai/gpt-oss-20b",     
    temperature=0.7,
).bind(response_format={"type": "json_object"})



chain_champion = model_prompt_champion | model | parser_champion    
response_champion = chain_champion.invoke({"champ": "Viego"})

chain_counter = model_prompt_counter | model | parser_counter
response_counter = chain_counter.invoke({"name": response_champion['name']})



print(f"{response_champion['name']} - Vantagem: {response_champion['advantage']}")
print(f"\n Contra {response_champion['name']} posso usar: {response_counter['name']} - Vantagem: {response_counter['advantage']}")
