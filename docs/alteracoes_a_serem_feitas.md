# Alterações a serem Feitas - Migração de Langchain para CrewAI

## Visão Geral da Migração

Este documento detalha as alterações necessárias para transformar o sistema atual do Pakkun, baseado em Langchain, em uma arquitetura baseada em CrewAI com uma estrutura real de crew utilizando o componente `crewai.Kickoff`. Esta migração preservará a funcionalidade atual enquanto reorganiza a arquitetura para aproveitar as capacidades colaborativas de agentes dentro da estrutura CrewAI.

## Arquitetura Atual vs. Nova Arquitetura

### Arquitetura Atual (Langchain)
Atualmente, o Pakkun utiliza um único agente Langchain que tem acesso a múltiplas ferramentas:
- Um retriever para consulta à base de conhecimento local (ChromaDB)
- Python REPL para execução de código
- Brave Search e StackExchange para busca na web

Este agente único é responsável por interpretar perguntas, decidir quais ferramentas usar, e sintetizar respostas.

### Nova Arquitetura (CrewAI)
A nova arquitetura introduzirá uma crew de agentes especializados, cada um com responsabilidades específicas:

1. **Agente Leitor da Base**: Especialista em consultar a base ChromaDB e extrair informações relevantes
2. **Agente Interpretador de Contexto**: Especialista em entender e contextualizar as informações extraídas
3. **Agente Gerador de Resposta**: Especialista em criar respostas claras e úteis para o usuário

Esses agentes trabalharão de forma colaborativa através do `crewai.Kickoff` para resolver problemas complexos relacionados ao código.

## Detalhamento das Substituições e Modificações

### 1. Substituição da Estrutura de Agente Único

#### Antes (Langchain - `/src/agent.py`):
```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools import python_repl, retriever_tool, brave_tool, stackexchange_tool
from embeddings import llm

agent_prompt = ChatPromptTemplate(
    input_variables=["input"],
    messages=[
        (
            "system",
            """Você é Pakkun, um assistente especialista em desenvolvimento de software...
            # ... resto do prompt do sistema
            """
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", output_key='output')

agent = create_tool_calling_agent(
    llm, [python_repl, retriever_tool, brave_tool, stackexchange_tool], agent_prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[python_repl, retriever_tool, brave_tool, stackexchange_tool],
    memory=memory,
    verbose=True,
    max_iterations=15,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

def get_agent_executor():
    return agent_executor
```

#### Depois (CrewAI - `/src/crew_setup.py`):
```python
from crewai import Agent, Task, Crew, Process
from crewai.kickoff import Kickoff
from tools import python_repl, brave_tool, stackexchange_tool
from embeddings import llm, retriever
from config import LLM_MODEL

# Definição dos agentes especializados
reader_agent = Agent(
    role="Leitor da Base de Código",
    goal="Extrair informações precisas e relevantes da base de código indexada",
    backstory="""Você é um especialista em recuperação de informações com profundo conhecimento
    de sistemas de código. Sua habilidade em buscar e identificar trechos relevantes de código
    é inigualável.""",
    llm=llm,
    verbose=True,
    tools=[retriever.as_tool()] 
)

interpreter_agent = Agent(
    role="Interpretador de Contexto",
    goal="Analisar e contextualizar informações técnicas relacionadas ao código",
    backstory="""Você é um engenheiro de software experiente com profunda compreensão de 
    múltiplas linguagens e frameworks. Você tem a capacidade de entender rapidamente estruturas
    de código e explicar conceitos técnicos complexos.""",
    llm=llm,
    verbose=True,
    tools=[python_repl]
)

response_agent = Agent(
    role="Gerador de Respostas",
    goal="Criar respostas claras, precisas e úteis para questões técnicas",
    backstory="""Você é um comunicador técnico excepcional, capaz de transformar
    informações técnicas complexas em explicações claras e acionáveis para desenvolvedores
    de todos os níveis de experiência.""",
    llm=llm,
    verbose=True,
    tools=[brave_tool, stackexchange_tool]
)

# Definição das tasks para cada agente
def create_tasks(query, chat_history=None):
    # Task para o agente leitor
    reader_task = Task(
        description=f"""
        Consulte a base de conhecimento para encontrar informações relevantes sobre:
        "{query}"
        
        Histórico de conversa para contexto:
        {chat_history if chat_history else "Sem histórico prévio"}
        
        Seja detalhado em sua busca e extraia todos os trechos de código e informações
        relevantes para responder à pergunta.
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
        
        Use as tags <think> e </think> para incluir seu raciocínio, que será exibido
        no expansor de detalhes.
        """,
        agent=response_agent,
        expected_output="Resposta final formatada para o usuário",
        context=[interpreter_task]
    )
    
    return [reader_task, interpreter_task, response_task]

# Configuração da crew
def get_crew(query, chat_history=None):
    tasks = create_tasks(query, chat_history)
    
    crew = Crew(
        agents=[reader_agent, interpreter_agent, response_agent],
        tasks=tasks,
        verbose=True,
        process=Process.sequential,
        memory=True,
    )
    
    return crew

# Kickoff para iniciar o fluxo
def get_crew_executor():
    return Kickoff(
        max_iterations=15,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
```

### 2. Adaptação da Interface Streamlit

#### Antes (app.py):
```python
import streamlit as st
from agent import get_agent_executor

# ... resto do código app.py ...

agent_executor = get_agent_executor()

if question := st.chat_input("Digite sua pergunta sobre o código"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        render_message(question)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = agent_executor.invoke({"input": question}).get(
                "output", "Não consegui processar sua pergunta."
            )
            render_message(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
```

#### Depois (app.py modificado):
```python
import streamlit as st
from crew_setup import get_crew, get_crew_executor

# ... resto do código app.py ...

crew_executor = get_crew_executor()

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
            result = crew_executor.kickoff(crew=crew)
            
            # Última task contém a resposta final
            response = result.get_output()
            if not response:
                response = "Não consegui processar sua pergunta."
            
            render_message(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
```

### 3. Ferramentas e Embeddings

As ferramentas atuais precisam ser refatoradas para funcionar com CrewAI, pois a API é ligeiramente diferente:

#### Antes (tools.py):
```python
from langchain_core.tools import Tool
# ... importações ...

python_repl = Tool(
    name="python_repl",
    description="Utilize para validar hipóteses...",
    func=PythonREPL().run,
    verbose=True,
)

# ... outras ferramentas ...
```

#### Depois (tools.py refatorado):
```python
from crewai.tools import Tool, BaseTool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import BraveSearch
from langchain_community.utilities import StackExchangeAPIWrapper
from config import BRAVE_API_KEY
from typing import Any

class PythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = "Utilize para validar hipóteses, executar pequenos trechos de código Python..."
    
    def _run(self, code: str) -> Any:
        repl = PythonREPL()
        return repl.run(code)

python_repl = PythonREPLTool()

# Analogamente para outras ferramentas
# ...
```

## Descrição dos Agentes da Crew

### 1. Agente Leitor da Base

**Função Principal**: Este agente é especializado em consultar a base de conhecimento ChromaDB de forma eficiente.

**Habilidades**:
- Especializado em recuperação de informações (RAG)
- Profundo conhecimento da estrutura de metadados do ChromaDB
- Capacidade de formular consultas precisas para extrair informações relevantes
- Entendimento da estrutura de projetos de software

**Ferramentas**: Utiliza principalmente o `retriever_tool` para consultar a base ChromaDB.

### 2. Agente Interpretador de Contexto

**Função Principal**: Este agente analisa e contextualiza as informações técnicas recuperadas pelo Agente Leitor.

**Habilidades**:
- Compreensão técnica profunda de múltiplas linguagens e frameworks
- Capacidade de conectar diferentes componentes e entender suas relações
- Habilidade para testar e validar hipóteses sobre o código
- Experiência em explicar conceitos técnicos complexos

**Ferramentas**: Utiliza principalmente o `python_repl` para testar hipóteses e validar entendimentos.

### 3. Agente Gerador de Resposta

**Função Principal**: Este agente sintetiza as informações analisadas em respostas claras e úteis para o usuário.

**Habilidades**:
- Excelente comunicação técnica
- Capacidade de simplificar conceitos complexos sem perder precisão
- Conhecimento para enriquecer respostas com recursos externos quando necessário
- Expertise em formatação e estruturação de conteúdo técnico

**Ferramentas**: Utiliza `brave_tool` e `stackexchange_tool` para complementar respostas com informações externas quando necessário.

## Organização das Tasks via Kickoff

O componente `crewai.Kickoff` será utilizado para gerenciar o fluxo das tarefas entre os agentes. O fluxo de trabalho será:

1. O `Kickoff` recebe a pergunta do usuário e o histórico de conversa
2. Cria uma instância da crew com os três agentes e suas tarefas associadas
3. Inicia o processo sequencial, onde:
   - O Agente Leitor recupera informações relevantes da base
   - O Agente Interpretador analisa essas informações e as contextualiza
   - O Agente Gerador cria a resposta final para o usuário
4. Retorna o resultado final para ser exibido na interface Streamlit

Este fluxo preserva a experiência de usuário do sistema atual, mas com um processamento interno mais especializado e colaborativo.
