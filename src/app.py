import streamlit as st
from crew_setup import get_crew, run_crew


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
            formatted_history = []
            if len(st.session_state.chat_history) > 1:
                for msg in st.session_state.chat_history[:-1]:
                    formatted_history.append(f"{msg['role']}: {msg['content']}")
                formatted_history = "\n".join(formatted_history)
            else:
                formatted_history = None

            crew = get_crew(question, formatted_history)

            try:
                result = run_crew(crew)
                response = result.last_task_output
                if not response:
                    response = "Não consegui processar sua pergunta."
            except Exception as e:
                response = f"Ocorreu um erro ao processar sua pergunta: {str(e)}"

            render_message(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
