# 🐕 Pakkun Project - Assistente de Código com RAG Avançado

Pakkun é um assistente inteligente de código que utiliza técnicas avançadas de RAG (**Retrieval Augmented Generation**) para fornecer respostas precisas sobre sua base de código. O projeto utiliza **ChromaDB** como banco de dados vetorial local com **coleções individuais por arquivo** e **Streamlit** para a interface interativa.

## 📊 Funcionalidades

- **Indexação por Arquivo**: Cada arquivo tem sua própria coleção para buscas mais precisas e contextualizadas
- **Parsers Específicos por Linguagem**: Processamento especializado para Python e JavaScript/TypeScript
- **Chunking Otimizado**: Chunks maiores e com sobreposição para melhor contexto e compreensão
- **Busca Semântica Avançada**: Múltiplas estratégias de busca com priorização inteligente
- **Extração Estrutural**: Detecção sofisticada de classes, funções, componentes React e hooks
- **Interface Intuitiva**: Interface de chat amigável com formatação melhorada e visualização de raciocínio

## 📋 Pré-requisitos

- **Python** 3.8+
- **Streamlit** (https://streamlit.io/)
- **ChromaDB** para armazenamento de vetores
- **Bibliotecas listadas em** `requirements.txt`
- **Chave API para o Groq** (ou outro LLM compatível)
- **Chave API para o Brave** 

## 🚀 Instalação

### 1️⃣ Clonar o repositório:

```sh
git clone https://github.com/joaoVittorDevv/PakkunProject.git
cd PakkunProject
```

### 2️⃣ Configurar ambiente virtual:

```sh
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3️⃣ Instalar dependências:

```sh
pip install -r requirements.txt
```

### 4️⃣ Executar o projeto:

```sh
streamlit run src/app.py
```

A aplicação abrirá automaticamente em seu navegador padrão.

## 📂 Estrutura do Projeto

```
PakkunProject/
├── scripts/          # Scripts auxiliares
│   └── code_indexer.py        # Indexador dos códigos
├── src/              # Código fonte principal
│   └── agent.py        # Agente
│   └── app.py        # Aplicação Streamlit
│   └── config.py        # Configurações do projeto
│   └── embeddings.py        # Criador dos embeddings e do retriever
│   └── tools.py        # Tools disponíveis para o agent
├── .gitignore        # Arquivos ignorados pelo Git
├── README.md         # Documentação
└── requirements.txt  # Dependências
```

## 📝 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

