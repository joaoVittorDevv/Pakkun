# Documentação do Sistema Atual - Pakkun

## Visão Geral

O Pakkun é um assistente de código que utiliza técnicas de RAG (Retrieval Augmented Generation) para fornecer respostas precisas sobre uma base de código. O sistema é construído com Langchain para orquestração do agente e utiliza ChromaDB como banco de dados vetorial local para armazenar e recuperar documentos de código.

## Arquivos e Componentes do Sistema

### `/scripts/code_indexer.py`

**Função Principal**: Responsável pela indexação dos arquivos de código que serão consultados pelo assistente.

**Tecnologias/Bibliotecas**:
- langchain_huggingface (HuggingFaceEmbeddings)
- langchain_chroma (Chroma)
- torch
- chardet (para detecção de encoding)

**Relações**:
- Cria e povoa a base ChromaDB utilizada pelo componente `embeddings.py`
- Define os parâmetros de chunking (tamanho e sobreposição)
- Configura o modelo de embeddings utilizado

**Contribuição para o Fluxo**: Este script prepara a base de conhecimento do sistema, indexando arquivos de código com metadados relevantes. É executado antes do uso da aplicação principal para criar os vetores de embeddings necessários para o retrieval.

### `/src/agent.py`

**Função Principal**: Define o agente Langchain e suas ferramentas, responsável por interpretar perguntas e gerar respostas.

**Tecnologias/Bibliotecas**:
- langchain.prompts (ChatPromptTemplate)
- langchain.agents (create_tool_calling_agent, AgentExecutor)
- langchain.memory (ConversationBufferMemory)

**Relações**:
- Importa ferramentas de `tools.py` 
- Utiliza o LLM definido em `embeddings.py`
- É importado por `app.py` para fornecer funcionalidade ao front-end

**Contribuição para o Fluxo**: Cria o agente de IA central do sistema, configurando seu prompt de sistema, memória de conversação e conjunto de ferramentas. Este agente é quem realiza o processamento das perguntas do usuário e a orquestração das ferramentas de busca e execução.

### `/src/app.py`

**Função Principal**: Interface de usuário do sistema via Streamlit.

**Tecnologias/Bibliotecas**:
- streamlit

**Relações**:
- Importa o agente executor de `agent.py`

**Contribuição para o Fluxo**: Fornece a interface gráfica para interação dos usuários com o agente Pakkun. Gerencia o histórico de conversação, renderiza respostas formatadas (incluindo expansores para raciocínio detalhado) e encaminha perguntas para processamento pelo agente.

### `/src/config.py`

**Função Principal**: Centraliza as configurações do sistema.

**Tecnologias/Bibliotecas**:
- decouple
- dotenv
- torch

**Relações**:
- É importado por `embeddings.py` e `tools.py`

**Contribuição para o Fluxo**: Define constantes e configurações essenciais, como modelo de embeddings, modelo LLM, dispositivo de processamento (CPU/CUDA) e chaves de API.

### `/src/embeddings.py`

**Função Principal**: Configura o sistema de embeddings e retriever para consulta de documentos.

**Tecnologias/Bibliotecas**:
- langchain_huggingface (HuggingFaceEmbeddings)
- langchain_chroma (Chroma)
- langchain.retrievers.self_query (SelfQueryRetriever)
- langchain_groq (ChatGroq)

**Relações**:
- Importa configurações de `config.py`
- Exporta o LLM usado por `agent.py`
- Exporta o retriever usado por `tools.py`

**Contribuição para o Fluxo**: Configura o modelo de embeddings, a conexão com o ChromaDB e o sistema de recuperação (retriever) com capacidade de auto-query. Define também o LLM utilizado pelo sistema, neste caso, usando o serviço Groq.

### `/src/tools.py`

**Função Principal**: Define ferramentas disponíveis para o agente.

**Tecnologias/Bibliotecas**:
- langchain_core.tools (Tool)
- langchain_experimental.utilities (PythonREPL)
- langchain_community.tools (BraveSearch)
- langchain_community.utilities (StackExchangeAPIWrapper)

**Relações**:
- Importa configurações de `config.py`
- Importa retriever de `embeddings.py`
- Exporta ferramentas usadas por `agent.py`

**Contribuição para o Fluxo**: Expõe ferramentas que o agente pode utilizar durante a interação com o usuário, incluindo:
- Execução de código Python (python_repl)
- Consulta à base de conhecimento indexada (retriever_tool)
- Busca na internet via Brave Search API (brave_tool)
- Consulta ao StackExchange (stackexchange_tool)

## Fluxo de Execução do Sistema

1. **Preparação de Dados**:
   - O `code_indexer.py` é executado para indexar arquivos de código
   - Os embeddings são gerados e armazenados no banco ChromaDB

2. **Inicialização do Sistema**:
   - `app.py` importa o agente executor de `agent.py`
   - Interface Streamlit é iniciada e apresentada ao usuário

3. **Interação**:
   - Usuário digita uma pergunta na interface
   - `app.py` encaminha a pergunta para o executor do agente
   - O agente (`agent.py`) processa a pergunta e determina quais ferramentas utilizar
   - Ferramentas em `tools.py` são chamadas conforme necessário (retriever para consulta à base, brave para busca web, etc.)
   - Resposta é gerada pelo agente e retornada para `app.py`
   - Interface Streamlit exibe a resposta ao usuário

4. **Memória de Conversação**:
   - O sistema mantém o contexto da conversa utilizando o `ConversationBufferMemory` do Langchain
   - Histórico é utilizado em perguntas subsequentes para manter coerência
