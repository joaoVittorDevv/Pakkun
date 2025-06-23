from typing import Any
import requests
from crewai.tools import tool
from config import BRAVE_API_KEY


class PythonREPL:
    def __init__(self) -> None:
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}

    def run(self, code: str) -> str:
        try:
            exec(code, self.globals, self.locals)
            lines = code.strip().split("\n")
            if lines:
                last = lines[-1].strip()
                if (
                    not last.startswith("def ")
                    and not last.startswith("class ")
                    and "=" not in last
                    and not last.startswith("import ")
                    and not last.startswith("from ")
                ):
                    try:
                        return str(eval(last, self.globals, self.locals))
                    except Exception:
                        pass
            return "Comando executado com sucesso."
        except Exception as exc:
            return f"Erro ao executar código: {exc}"


python_repl_instance = PythonREPL()


@tool("python_repl")
def python_repl(code: str) -> str:
    """Executa pequenos trechos de código Python"""
    return python_repl_instance.run(code)


class BraveSearch:
    def __init__(self, api_key: str = BRAVE_API_KEY) -> None:
        self.api_key = api_key
        self.api_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, count: int = 3) -> str:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": count}
        try:
            resp = requests.get(self.api_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = []
            if "web" in data and "results" in data["web"]:
                for item in data["web"]["results"]:
                    results.append(
                        f"Título: {item.get('title', 'Sem título')}\n"
                        f"Descrição: {item.get('description', 'Sem descrição')}\n"
                        f"URL: {item.get('url', 'Sem URL')}"
                    )
            return "\n\n".join(results) if results else "Nenhum resultado encontrado."
        except Exception as exc:
            return f"Erro ao realizar busca: {exc}"


brave_search_instance = BraveSearch()


@tool("brave_tool")
def brave_tool(query: str) -> str:
    """Busca informações atualizadas na web"""
    return brave_search_instance.search(query)


class StackExchangeSearch:
    def __init__(self) -> None:
        self.api_url = "https://api.stackexchange.com/2.3/search/advanced"

    def search(self, query: str) -> str:
        params = {
            "q": query,
            "site": "stackoverflow",
            "order": "desc",
            "sort": "relevance",
            "accepted": "True",
            "filter": "withbody",
            "pagesize": 3,
        }
        try:
            resp = requests.get(self.api_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if "items" not in data or not data["items"]:
                return "Nenhum resultado encontrado no StackExchange."
            results = []
            for item in data["items"]:
                results.append(
                    f"Pergunta: {item.get('title')}\n"
                    f"Score: {item.get('score')}\n"
                    f"Link: {item.get('link')}\n"
                    f"Resumo: {item.get('excerpt', '')[:300]}..."
                )
            return "\n\n".join(results)
        except Exception as exc:
            return f"Erro ao buscar no StackExchange: {exc}"


stackexchange_instance = StackExchangeSearch()


@tool("stackexchange_tool")
def stackexchange_tool(query: str) -> str:
    """Busca respostas no StackExchange"""
    return stackexchange_instance.search(query)
