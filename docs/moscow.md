# MoSCoW - Priorização da Refatoração Langchain para CrewAI

Este documento utiliza o framework MoSCoW para priorizar os elementos da refatoração do sistema Pakkun de Langchain para CrewAI. Esta abordagem ajudará a equipe a focar nos aspectos mais críticos do projeto primeiro, garantindo uma transição bem-sucedida.

## Must Have (Deve ter)

Estes são os requisitos essenciais sem os quais a refatoração não pode ser considerada bem-sucedida:

1. **Substituir o Agente Langchain por uma CrewAI**
   - Implementar a estrutura básica da CrewAI com os três agentes especializados
   - Configurar o componente crewai.Kickoff para coordenar o fluxo de trabalho entre os agentes
   - Garantir que a crew possa processar perguntas e gerar respostas coerentes

2. **Manter a Interface Streamlit**
   - Adaptar o arquivo app.py para se integrar à nova arquitetura CrewAI
   - Preservar a experiência do usuário, incluindo a formatação de respostas com expandidores
   - Garantir que o histórico de conversa continue funcionando corretamente

3. **Criar Agentes Especializados**
   - Implementar o Agente Leitor da Base para consultas ao ChromaDB
   - Implementar o Agente Interpretador de Contexto para análise do código
   - Implementar o Agente Gerador de Resposta para criação de conteúdo para o usuário

4. **Adaptar Integração com Base ChromaDB**
   - Garantir que o sistema CrewAI possa acessar a base ChromaDB existente
   - Adaptar o retriever para funcionar no contexto do CrewAI
   - Manter a capacidade de busca vetorial precisa

5. **Manter Funcionalidades Essenciais**
   - Preservar capacidade de executar código Python através do Python REPL
   - Manter capacidade de buscar informações externas (Brave, StackExchange)
   - Preservar capacidade de formatação de resposta com tags `<think>` e `</think>`

## Should Have (Deveria ter)

Estes são requisitos importantes que agregam valor significativo, mas não impedem o lançamento inicial:

1. **Modularizar Responsabilidades dos Agentes**
   - Refinar os prompts de cada agente para maximizar sua especialização
   - Implementar mecanismos robustos de passagem de contexto entre agentes
   - Criar mecanismos para que os agentes solicitem informações adicionais quando necessário

2. **Documentação Técnica Detalhada**
   - Criar diagramas de fluxo que ilustrem o processo de trabalho da crew
   - Documentar a API de comunicação entre os componentes
   - Fornecer exemplos detalhados de como estender o sistema

3. **Refatoração das Ferramentas**
   - Adaptar todas as ferramentas existentes para o formato CrewAI
   - Modularizar as ferramentas para fácil reutilização entre agentes
   - Adicionar validação de entrada e saída para cada ferramenta

4. **Mecanismo de Comunicação entre Agentes**
   - Implementar protocolo estruturado para troca de informações entre agentes
   - Criar formatação padronizada para passagem de contexto
   - Documentar como adicionar novos agentes ao sistema

5. **Gestão de Estado da Conversa**
   - Implementar mecanismo robusto para manter o estado da conversa entre interações
   - Permitir que os agentes acessem e utilizem o histórico de conversa
   - Garantir persistência do contexto ao longo da sessão do usuário

## Could Have (Poderia ter)

Estes são requisitos desejáveis que melhorariam o sistema, mas não são críticos para a primeira versão:

1. **Logging Estruturado entre Agentes**
   - Implementar sistema de logging detalhado para rastrear a comunicação entre agentes
   - Criar visualização do processo de raciocínio entre agentes
   - Permitir ao usuário ver o "bastidores" do trabalho da crew

2. **Cache de Respostas**
   - Implementar sistema de cache para perguntas frequentes
   - Criar mecanismo para invalidar cache quando a base de código muda
   - Otimizar desempenho através de estratégias de caching inteligente

3. **Mecanismo de Feedback do Usuário**
   - Permitir que usuários avaliem respostas
   - Usar feedback para melhorar prompts e comportamento dos agentes
   - Implementar sistema de aprendizado incremental

4. **Interface para Configuração da Crew**
   - Permitir customização dos papéis dos agentes via interface
   - Criar painel para ajustar parâmetros da crew
   - Implementar diferentes perfis de crew para diferentes tipos de tarefas

5. **Métricas de Desempenho**
   - Implementar rastreamento de tempo de resposta por agente
   - Criar dashboards de qualidade das respostas
   - Monitorar uso de tokens e recursos

## Won't Have (Não terá por enquanto)

Estes requisitos foram explicitamente deixados de fora do escopo atual da refatoração:

1. **Testes Automatizados**
   - Framework completo de testes unitários e integração
   - Testes de regressão automatizados
   - CI/CD para validação automática

2. **Múltiplas Crews**
   - Capacidade de instanciar diferentes crews para diferentes tipos de tarefas
   - Sistema para rotear perguntas para crews especializadas
   - Mecanismo de colaboração entre crews distintas

3. **Processamento Paralelo**
   - Execução paralela de agentes para tarefas independentes
   - Distribuição de carga entre agentes
   - Escalonamento dinâmico baseado em demanda

4. **Sistema de Plugins**
   - Framework para adicionar novos agentes ou ferramentas via plugins
   - Marketplace de componentes
   - Sistema de versionamento de plugins

5. **Interfaceamento com Outros Serviços de IA**
   - Integração com múltiplos provedores de LLM
   - Sistema de fallback entre diferentes modelos
   - Mecanismo de avaliação comparativa entre diferentes LLMs
