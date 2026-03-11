# O Que é RAG (Retrieval-Augmented Generation)?

RAG, ou **Geração Aumentada de Recuperação**, é uma técnica que permite a um Modelo de Linguagem (LLM), como o GPT, acessar informações que não estavam em seu treinamento original. 

Imagine que o LLM é um estudante muito inteligente, mas que fez sua última prova há dois anos. Se você perguntar algo que aconteceu ontem, ele vai "alucinar" ou dizer que não sabe. O RAG é como dar a esse estudante um **livro de consulta** atualizado ou manuais específicos da sua empresa para que ele consulte antes de responder.

---

## Como o RAG Funciona? (O Fluxo Real)

O processo de RAG pode ser dividido em duas grandes fases:

### 1. Preparação dos Dados (Indexação)
Antes de perguntar algo, precisamos preparar o "conhecimento" que o modelo vai consultar:
1. **Carregamento**: Pegamos documentos (PDFs, sites, bancos de dados).
2. **Fragmentação (Splitting)**: Quebramos textos longos em pedaços menores (chunks), para que o modelo não se perca.
3. **Embeddings**: Transformamos esses textos em números (vetores) que representam o significado do texto.
4. **Vector Store**: Guardamos esses vetores em um banco de dados especial (como Chroma, Pinecone ou FAISS).

### 2. Recuperação e Resposta (Inference)
Quando o usuário faz uma pergunta:
1. **Consulta**: A pergunta do usuário também é transformada em um vetor (embedding).
2. **Busca Semântica**: O sistema procura no banco de vetores os "pedaços de texto" que são numericamente mais parecidos com a pergunta.
3. **Aumentação**: Colocamos esses pedaços de texto dentro do *prompt* (a instrução) para o LLM.
4. **Geração**: O modelo lê a pergunta **e** os pedaços de texto fornecidos, gerando uma resposta precisa baseada neles.

---

## RAG com LangChain

O **LangChain** é a ferramenta que "cola" todas essas partes de forma simples. Sem ele, você teria que lidar com APIs de vetores, lógica de busca e formatação de prompts manualmente.

### Componentes Principais do LangChain para RAG:

1. **Document Loaders**: Ferramentas para ler quase qualquer formato (`PyPDFLoader`, `WebBaseLoader`).
2. **Text Splitters**: Funções que dividem o texto de forma inteligente (ex: `RecursiveCharacterTextSplitter`).
3. **Embeddings**: Interfaces para modelos que geram os vetores (OpenAI, HuggingFace, Ollama).
4. **Vector Stores**: Onde os dados ficam guardados.
5. **Retriever**: O "garçom" que vai lá no banco de dados buscar o que você precisa.
6. **Chains (Cadeias)**: O fluxo que une o Retriever ao LLM.

### Exemplo de Código (Conceitual)

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Carregar o documento
loader = TextLoader("meu_documento.txt")
docs = loader.load()

# 2. Dividir o texto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Criar o Banco de Vetores (Embeddings + Vector Storage)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 4. Criar a "Corrente" de Resposta
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Perguntar!
resposta = qa_chain.invoke("Qual o resumo do documento?")
print(resposta)
```

---

## Por que usar RAG?

*   **Evita Alucinações**: O modelo responde com base em fatos fornecidos, não apenas no que ele "acha".
*   **Dados Privados**: Você pode usar documentos que nunca estiveram na internet.
*   **Atualização em Tempo Real**: Se o documento mudar, basta atualizar o banco de vetores, sem precisar treinar o modelo de novo.
