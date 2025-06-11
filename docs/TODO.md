# TODO: Roteiro Técnico para Refatoração de Langchain para CrewAI

Este documento detalha o passo a passo para refatorar o sistema Pakkun de uma arquitetura baseada em Langchain para uma estrutura de colaboração de agentes usando CrewAI. Siga este guia sequencial para realizar a migração completa.

## 1. Preparação do Ambiente

### 1.1 Remover Dependências do Langchain

```bash
pip uninstall langchain langchain-core langchain_experimental langchain_huggingface langchain_community langchain_chroma langchain_groq -y
```

### 1.2 Instalar CrewAI e Dependências Necessárias

```bash
pip install crewai chromadb chroma-hnswlib groq streamlit sentence-transformers python-decouple python-dotenv
```

### 1.3 Atualizar arquivo requirements.txt

```bash
cat > requirements.txt << EOF
crewai>=0.28.0
streamlit>=1.31.1
chromadb>=0.4.20
chroma-hnswlib>=0.7.3
sentence-transformers>=2.2.2
groq>=0.4.0
python-decouple>=3.8
python-dotenv>=1.0.0
requests>=2.31.0
typing-extensions>=4.8.0
chardet>=5.2.0
EOF
```

## 2. Criar Nova Estrutura de Arquivos

### 2.1 Criar novo arquivo para configuração da Crew

```bash
touch src/crew_setup.py
```

## 3. Implementação da Estrutura CrewAI

### 3.1 Refatorar Embedding e Retriever

O primeiro passo é adaptar o arquivo `embeddings.py` para compatibilidade com CrewAI, mantendo a funcionalidade existente:

```python
# Editar src/embeddings.py
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient, Documents, Embeddings
import chromadb.config
from groq import Groq
from crewai.tools.basetool import BaseTool
from config import EMBEDDINGS_MODEL, DEVICE, LLM_MODEL
import os
from typing import List, Optional, Any

# Configuração dos embeddings - mantida similar
embeddings_model = SentenceTransformer(
    EMBEDDINGS_MODEL,
    device=DEVICE
)

# Cliente ChromaDB
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("code_collection")

# Configuração do LLM com Groq - adaptada para uso direto
groq_client = Groq()
llm = groq_client

# Classe para o retriever compatível com CrewAI
class ChromaRetriever:
    def __init__(
        self, 
        collection_name: str = "code_collection",
        persist_directory: str = "./chroma_db",
        embedding_model: Any = embeddings_model,
        k: int = 5
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.k = k
        self.client = chroma_client
        self.collection = collection
    
    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        """Busca documentos relevantes baseados na query"""
        if k is None:
            k = self.k
            
        # Gerar embedding para a query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Realizar a busca
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatar resultados
        documents = []
        for i in range(len(results["documents"][0])):
            doc = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                "score": 1.0 - results["distances"][0][i] if results["distances"][0] else 0.0,
            }
            documents.append(doc)
        
        return documents
    
    def invoke(self, query: str) -> str:
        """Método que permite usar o retriever como ferramenta independente"""
        results = self.search(query)
        return "\n\n".join([
            f"**Documento de {doc['metadata'].get('full_path', 'Desconhecido')}**\n"
            f"Relevância: {doc['score']:.2f}\n\n{doc['content']}"
            for doc in results
        ])
    
    def as_tool(self) -> BaseTool:
        """Retorna o retriever como uma ferramenta compatível com CrewAI"""
        from crewai.tools import Tool
        
        return Tool(
            name="retriever_tool",
            description="Use essa ferramenta para buscar informações específicas dentro do contexto completo dos arquivos do projeto, incluindo Django, React, Docker, bancos de dados, testes e outras tecnologias associadas. Ideal para entender estruturas, padrões de projeto, exemplos práticos e para esclarecer dúvidas técnicas sobre o código existente.",
            func=self.invoke
        )

# Instanciar o retriever
retriever = ChromaRetriever()
```

### 3.2 Refatorar Ferramentas

Adaptar as ferramentas existentes para o formato CrewAI:

```python
# Editar src/tools.py
from crewai.tools import Tool
from typing import Any
import requests
import json
from config import BRAVE_API_KEY

# Implementação do Python REPL como ferramenta CrewAI
class PythonREPLTool:
    def __init__(self):
        self.locals = {}
        self.globals = {}
    
    def execute(self, code: str) -> str:
        """Execute código Python e retorne o resultado"""
        try:
            # Use exec para comandos e eval para expressões
            exec_result = exec(code, self.globals, self.locals)
            # Se exec não retornou nada (como atribuições), tente avaliar a última linha
            lines = code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                # Verifica se a última linha não é uma atribuição ou definição
                if (not last_line.startswith('def ') and 
                    not last_line.startswith('class ') and 
                    '=' not in last_line and 
                    not last_line.startswith('import ') and
                    not last_line.startswith('from ')):
                    try:
                        eval_result = eval(last_line, self.globals, self.locals)
                        return str(eval_result)
                    except:
                        pass
            
            # Se não conseguir avaliar a última linha, retorne o resultado do exec
            # (que provavelmente é None para comandos como atribuições)
            return str(exec_result) if exec_result is not None else "Comando executado com sucesso."
        except Exception as e:
            return f"Erro ao executar código: {str(e)}"

# Brave Search Tool
class BraveSearchTool:
    def __init__(self, api_key: str = BRAVE_API_KEY):
        self.api_key = api_key
        self.api_url = "https://api.search.brave.com/res/v1/web/search"
    
    def search(self, query: str, count: int = 3) -> str:
        """Realiza uma busca usando a API do Brave Search"""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": count
        }
        
        try:
            response = requests.get(
                self.api_url,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Formatar resultados
            formatted_results = []
            if "web" in results and "results" in results["web"]:
                for item in results["web"]["results"]:
                    formatted_results.append(
                        f"Título: {item.get('title', 'Sem título')}\n"
                        f"Descrição: {item.get('description', 'Sem descrição')}\n"
                        f"URL: {item.get('url', 'Sem URL')}\n"
                    )
            
            return "\n\n".join(formatted_results) if formatted_results else "Nenhum resultado encontrado."
            
        except Exception as e:
            return f"Erro ao realizar busca: {str(e)}"

# StackExchange Tool
class StackExchangeTool:
    def __init__(self):
        self.api_url = "https://api.stackexchange.com/2.3/search/advanced"
    
    def search(self, query: str) -> str:
        """Busca no StackExchange por perguntas relacionadas à query"""
        params = {
            "q": query,
            "site": "stackoverflow",
            "order": "desc",
            "sort": "relevance",
            "accepted": "True",  # Somente perguntas com respostas aceitas
            "filter": "withbody",  # Incluir corpo das perguntas e respostas
            "pagesize": 3
        }
        
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            
            results = response.json()
            
            if "items" not in results or not results["items"]:
                return "Nenhum resultado encontrado no StackExchange."
            
            formatted_results = []
            for item in results["items"]:
                formatted_results.append(
                    f"Pergunta: {item.get('title', 'Sem título')}\n"
                    f"Score: {item.get('score', 'N/A')}\n"
                    f"Link: {item.get('link', 'Sem link')}\n"
                    f"Resumo: {item.get('excerpt', 'Sem resumo')[:300]}...\n"
                )
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Erro ao buscar no StackExchange: {str(e)}"

# Instanciamento das ferramentas
python_repl_instance = PythonREPLTool()
brave_search_instance = BraveSearchTool()
stackexchange_instance = StackExchangeTool()

# Criação das ferramentas no formato CrewAI
python_repl = Tool(
    name="python_repl",
    description="Utilize para validar hipóteses, executar pequenos trechos de código Python, verificar rapidamente resultados de algoritmos ou cálculos e garantir que a resposta técnica esteja correta antes de fornecê-la ao usuário.",
    func=python_repl_instance.execute
)

brave_tool = Tool(
    name="brave_tool",
    description="Utilize essa ferramenta para validar informações diretamente da internet, como correções de bugs, documentação atualizada, exemplos de uso ou quaisquer dúvidas gerais não cobertas pelas outras ferramentas.",
    func=brave_search_instance.search
)

stackexchange_tool = Tool(
    name="stackexchange_tool",
    description="Utilize essa ferramenta para validar informações diretamente da resposta de outras pessoas, como correções de bugs, documentação atualizada, exemplos de uso ou quaisquer dúvidas gerais não cobertas pelas outras ferramentas.",
    func=stackexchange_instance.search
)
```

### 3.3 Implementar a Configuração da CrewAI

Criar o novo arquivo `crew_setup.py` para a estrutura de agentes:

```python
# Criar src/crew_setup.py
from crewai import Agent, Task, Crew, Process
from crewai.kickoff import Kickoff
from tools import python_repl, brave_tool, stackexchange_tool
from embeddings import llm, retriever
from typing import List, Optional, Dict

# Definição dos agentes especializados
reader_agent = Agent(
    role="Leitor da Base de Código",
    goal="Extrair informações precisas e relevantes da base de código indexada",
    backstory="""Você é um especialista em recuperação de informações com profundo conhecimento
    de sistemas de código. Sua habilidade em buscar e identificar trechos relevantes de código
    é inigualável. Você consegue encontrar exatamente as partes do código que são relevantes
    para qualquer consulta e apresentá-las de forma organizada.""",
    verbose=True,
    allow_delegation=False,
    tools=[retriever.as_tool()]
)

interpreter_agent = Agent(
    role="Interpretador de Contexto",
    goal="Analisar e contextualizar informações técnicas relacionadas ao código",
    backstory="""Você é um engenheiro de software experiente com profunda compreensão de 
    múltiplas linguagens e frameworks. Você tem a capacidade de entender rapidamente estruturas
    de código e explicar conceitos técnicos complexos. Sua especialidade é conectar pedaços
    de informação e apresentar uma visão holística do código.""",
    verbose=True,
    allow_delegation=False,
    tools=[python_repl]
)

response_agent = Agent(
    role="Gerador de Respostas",
    goal="Criar respostas claras, precisas e úteis para questões técnicas",
    backstory="""Você é um comunicador técnico excepcional, capaz de transformar
    informações técnicas complexas em explicações claras e acionáveis para desenvolvedores
    de todos os níveis de experiência. Você estrutura respostas de forma lógica e 
    inclui exemplos relevantes quando necessário.""",
    verbose=True,
    allow_delegation=False,
    tools=[brave_tool, stackexchange_tool]
)

# Definição das tasks para cada agente
def create_tasks(query: str, chat_history: Optional[str] = None) -> List[Task]:
    # Task para o agente leitor
    reader_task = Task(
        description=f"""
        Consulte a base de conhecimento para encontrar informações relevantes sobre:
        "{query}"
        
        Histórico de conversa para contexto:
        {chat_history if chat_history else "Sem histórico prévio"}
        
        Seja detalhado em sua busca e extraia todos os trechos de código e informações
        relevantes para responder à pergunta. Utilize o retriever_tool para consultar
        a base de conhecimento.
        
        Formate sua resposta de forma clara, destacando os arquivos relevantes e
        os trechos de código importantes.
        """,
        agent=reader_agent,
        expected_output="Informações detalhadas da base de código que respondem à consulta",
    )
    
    # Task para o agente interpretador
    interpreter_task = Task(
        description=f"""
        Analise as informações recuperadas sobre:
        "{query}"
        
        Contextualize essas informações e execute testes ou validações quando necessário.
        Explique como os trechos de código funcionam no contexto mais amplo do projeto.
        Se preciso, teste hipóteses usando o Python REPL.
        
        Sua análise deve:
        1. Explicar o propósito do código
        2. Descrever como ele se integra ao restante do sistema
        3. Identificar padrões ou práticas notáveis
        4. Validar seu funcionamento quando apropriado
        """,
        agent=interpreter_agent,
        expected_output="Análise contextualizada do código com explicações técnicas detalhadas",
        context=[reader_task]
    )
    
    # Task para o agente de resposta
    response_task = Task(
        description=f"""
        Com base nas informações analisadas, crie uma resposta completa e clara para:
        "{query}"
        
        Sua resposta deve:
        1. Ser tecnicamente precisa
        2. Incluir exemplos relevantes
        3. Ser estruturada de forma lógica
        4. Usar recursos adicionais da web quando necessário para complementar
        
        IMPORTANTE: Use as tags <think> e </think> para incluir seu raciocínio,
        que será exibido no expansor de detalhes na interface do usuário.
        
        Formato esperado:
        [Resposta direta e concisa]
        
        <think>
        [Explicação detalhada do raciocínio, análise e processo]
        </think>
        
        [Conclusão ou informações adicionais]
        """,
        agent=response_agent,
        expected_output="Resposta final formatada para o usuário",
        context=[interpreter_task]
    )
    
    return [reader_task, interpreter_task, response_task]

# Configuração da crew
def get_crew(query: str, chat_history: Optional[str] = None) -> Crew:
    tasks = create_tasks(query, chat_history)
    
    crew = Crew(
        agents=[reader_agent, interpreter_agent, response_agent],
        tasks=tasks,
        verbose=True,
        process=Process.sequential,
    )
    
    return crew

# Kickoff para iniciar o fluxo
def get_crew_executor() -> Kickoff:
    return Kickoff(
        max_iterations=15,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
```

### 3.4 Adaptar a Interface Streamlit (app.py)

```python
# Editar src/app.py
import streamlit as st
from crew_setup import get_crew, get_crew_executor

st.set_page_config("Pakkun - Assistente de Código", "🐕", "centered")

with st.sidebar:
    st.title("🐕 Pakkun")
    st.markdown(
        """
        ### Como usar:
        - Perguntas específicas sobre código indexado
        - Explicações sobre código
        - Recomendações e boas práticas
    """
    )
    if st.button("Limpar conversa"):
        st.session_state.clear()
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": """
        Oi! Eu sou o Pakkun, como posso ajudar com seu código?

        Para que eu possa te ajudar melhor, você pode me pedir explicitamente para usar algumas ferramentas:

        - Se quiser que eu consulte códigos, arquivos ou documentos específicos do projeto, solicite diretamente que eu utilize a ferramenta `retriever_tool`. Exemplos:
          - "Use a retriever_tool para consultar o arquivo settings.py do Django"
          - "Busque no retriever_tool como foi feita a autenticação JWT no projeto React"

        - Caso precise que eu valide ideias, execute testes rápidos ou rode trechos de código Python, solicite o uso da ferramenta `python_repl`. Exemplos:
          - "Execute com python_repl uma função que valida esse regex"
          - "Use python_repl para testar rapidamente este trecho de código"

        - Para buscas atualizadas diretamente da internet, como documentações, correções de bugs ou exemplos externos, solicite o uso da ferramenta `brave_tool`. Exemplos:
          - "Use brave_tool para consultar a documentação mais recente do Django REST Framework"
          - "Consulte com brave_tool exemplos recentes de implementação do Zustand no React"

        - Caso precise buscar soluções técnicas comuns, exemplos práticos ou dúvidas respondidas pela comunidade, peça que eu utilize a ferramenta `stackexchange_tool`. Exemplos:
          - "Busque com stackexchange_tool como resolver esse erro do Docker"
          - "Use stackexchange_tool para encontrar exemplos de queries complexas no Django ORM"

        Lembre-se de solicitar explicitamente as ferramentas, pois não consigo acioná-las automaticamente!

        Estou aqui pra te ajudar com clareza e de forma amigável! 😉
        """,
        }
    ]

crew_executor = get_crew_executor()

def render_message(content):
    if "<think>" in content:
        prefix, _, rest = content.partition("<think>")
        think_content, _, suffix = rest.partition("</think>")
        if prefix.strip():
            st.markdown(prefix.strip())
        with st.expander("Detalhes 🧠"):
            st.info(think_content.strip())
        if suffix.strip():
            st.markdown(suffix.strip())
    else:
        st.markdown(content)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        render_message(msg["content"])

if question := st.chat_input("Digite sua pergunta sobre o código"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_message(question)

    with st.chat_message("assistant"):
        with st.spinner("A crew está trabalhando na sua pergunta..."):
            # Converter histórico de chat para formato utilizável pela crew
            formatted_history = []
            if len(st.session_state.chat_history) > 1:  # Se há histórico além da pergunta atual
                for msg in st.session_state.chat_history[:-1]:  # Exclui a pergunta atual
                    formatted_history.append(f"{msg['role']}: {msg['content']}")
                formatted_history = "\n".join(formatted_history)
            else:
                formatted_history = None
            
            # Criar crew com base na pergunta e histórico
            crew = get_crew(question, formatted_history)
            
            # Executar a crew com o kickoff
            try:
                result = crew_executor.kickoff(crew=crew)
                
                # A última task contém a resposta final
                response = result.last_task_output
                if not response:
                    response = "Não consegui processar sua pergunta."
            except Exception as e:
                response = f"Ocorreu um erro ao processar sua pergunta: {str(e)}"
            
            render_message(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
```

## 4. Atualizar `config.py`

Adaptar o arquivo de configuração para os novos requisitos do CrewAI:

```python
# Editar src/config.py
from decouple import config
from dotenv import load_dotenv
import torch

load_dotenv()

# Configurações mantidas do sistema antigo
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1-distill-llama-70b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# APIs
BRAVE_API_KEY = config("BRAVE_API_KEY")
GROQ_API_KEY = config("GROQ_API_KEY", default=None)

# Configurações específicas para CrewAI
MAX_ITERATIONS = 15
CREW_VERBOSE = True
```

## 5. Atualizar Arquivos de Inicialização

Garantir que o arquivo `__init__.py` esteja atualizado:

```python
# Editar src/__init__.py se necessário
# Normalmente pode estar vazio, mas assegure-se de que ele existe
```

## 6. Testar o Sistema

### 6.1 Teste Inicial e Verificação

```bash
streamlit run src/app.py
```

Verifique se a integração está funcionando corretamente e se a crew responde às perguntas de maneira adequada.

## 7. Aprimoramentos Futuros

Após a migração bem-sucedida, considere implementar estas melhorias:

1. **Melhorar os Prompts dos Agentes**:
   - Refinar os prompts de cada agente para maximizar sua especialização
   - Melhorar as instruções para os agentes sobre como formatar as respostas

2. **Adicionar Logs Estruturados**:
   - Implementar um sistema de logs para acompanhar o trabalho da crew
   - Adicionar visualizações dos passos intermediários

3. **Caching de Respostas**:
   - Desenvolver sistema de cache para perguntas frequentes
   - Implementar invalidação inteligente de cache

## Referências

- [Documentação CrewAI](https://crewai.io/docs/overview/first-crew)
- [Documentação Kickoff](https://crewai.io/docs/core-building-blocks/kickoff)
- [Guia MoSCoW (docs/moscow.md)](/home/jao/PycharmProjectsPro/Pakkun/docs/moscow.md)
- [Documentação do Sistema Atual (docs/documentacao_do_sistema_atual.md)](/home/jao/PycharmProjectsPro/Pakkun/docs/documentacao_do_sistema_atual.md)
- [Alterações a Serem Feitas (docs/alteracoes_a_serem_feitas.md)](/home/jao/PycharmProjectsPro/Pakkun/docs/alteracoes_a_serem_feitas.md)
