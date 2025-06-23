from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from tools import python_repl, brave_tool, stackexchange_tool
from embeddings import llm, retriever
from config import MAX_ITERATIONS, CREW_VERBOSE


# Agents definitions
reader_agent = Agent(
    role="Leitor da Base de Código",
    goal="Extrair informações precisas e relevantes da base de código indexada",
    backstory=(
        "Você é um especialista em recuperação de informações com profundo "
        "conhecimento de sistemas de código. Sua habilidade em buscar trechos "
        "relevantes é inigualável."
    ),
    llm=llm,
    verbose=CREW_VERBOSE,
    allow_delegation=False,
    tools=[retriever.as_tool()],
)

interpreter_agent = Agent(
    role="Interpretador de Contexto",
    goal="Analisar e contextualizar informações técnicas relacionadas ao código",
    backstory=(
        "Você é um engenheiro de software experiente com profunda compreensão de "
        "múltiplas linguagens e frameworks."
    ),
    llm=llm,
    verbose=CREW_VERBOSE,
    allow_delegation=False,
    tools=[python_repl],
)

response_agent = Agent(
    role="Gerador de Respostas",
    goal="Criar respostas claras, precisas e úteis para questões técnicas",
    backstory=(
        "Você é um comunicador técnico capaz de transformar informações complexas "
        "em explicações claras e acionáveis."
    ),
    llm=llm,
    verbose=CREW_VERBOSE,
    allow_delegation=False,
    tools=[brave_tool, stackexchange_tool],
)


def create_tasks(query: str, chat_history: Optional[str] = None) -> List[Task]:
    reader_task = Task(
        description=f"""
        Consulte a base de conhecimento para encontrar informações relevantes sobre:
        "{query}"

        Histórico de conversa para contexto:
        {chat_history if chat_history else 'Sem histórico prévio'}

        Seja detalhado em sua busca e extraia todos os trechos de código e informações
        relevantes para responder à pergunta. Utilize o retriever_tool para consultar
        a base de conhecimento.
        """,
        agent=reader_agent,
        expected_output="Informações detalhadas da base de código que respondem à consulta",
    )

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
        context=[reader_task],
    )

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
        """,
        agent=response_agent,
        expected_output="Resposta final formatada para o usuário",
        context=[interpreter_task],
    )

    return [reader_task, interpreter_task, response_task]


def get_crew(query: str, chat_history: Optional[str] = None) -> Crew:
    tasks = create_tasks(query, chat_history)
    crew = Crew(
        agents=[reader_agent, interpreter_agent, response_agent],
        tasks=tasks,
        verbose=CREW_VERBOSE,
        process=Process.sequential,
    )
    return crew


def run_crew(crew: Crew):
    return crew.kickoff(
        max_iterations=MAX_ITERATIONS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
