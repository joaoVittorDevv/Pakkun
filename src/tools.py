from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import BraveSearch
from langchain_community.utilities import StackExchangeAPIWrapper
from config import BRAVE_API_KEY
from embeddings import retriever

python_repl = Tool(
    name="python_repl",
    description="Utilize para validar hipóteses, executar pequenos trechos de código Python, verificar rapidamente resultados de algoritmos ou cálculos e garantir que a resposta técnica esteja correta antes de fornecê-la ao usuário.",
    func=PythonREPL().run,
    verbose=True,
)

retriever_tool = Tool(
    name="retriever_tool",
    description="Use essa ferramenta para buscar informações específicas dentro do contexto completo dos arquivos do projeto, incluindo Django, React, Docker, bancos de dados, testes e outras tecnologias associadas. Ideal para entender estruturas, padrões de projeto, exemplos práticos e para esclarecer dúvidas técnicas sobre o código existente.",
    func=lambda query: "\n".join(doc.page_content for doc in retriever.invoke(query)),
    verbose=True,
)

brave_search = BraveSearch.from_api_key(
    api_key=BRAVE_API_KEY, search_kwargs={"count": 3}
)
brave_tool = Tool(
    name="brave_tool",
    description="Utilize essa ferramenta para validar informações diretamente da internet, como correções de bugs, documentação atualizada, exemplos de uso ou quaisquer dúvidas gerais não cobertas pelas outras ferramentas.",
    func=lambda query: brave_search.run(query),
    verbose=True,
)

stackexchange = StackExchangeAPIWrapper()
stackexchange_tool = Tool(
    name="stackexchange_tool",
    description="Utilize essa ferramenta para validar informações diretamente da resposta de outras pessoas, como correções de bugs, documentação atualizada, exemplos de uso ou quaisquer dúvidas gerais não cobertas pelas outras ferramentas.",
    func=lambda query: stackexchange.run(query),
    verbose=True,
)
