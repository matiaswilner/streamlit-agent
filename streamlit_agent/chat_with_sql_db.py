import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# User inputs
radio_opt = ["Use sample database - Chinook.db", "Connect to your SQL database"]
selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
if radio_opt.index(selected_opt) == 1:
    db_uri = st.sidebar.text_input(
        label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
    )
else:
    db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
    db_uri = f"sqlite:////{db_filepath}"

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
)

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Setup agent
# llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)


db = configure_db(db_uri)

few_shots = {
    "List all photographers from Uruguay.": 'SELECT * FROM "user" u LEFT JOIN users_roles ur ON u.id = ur."userId" LEFT JOIN role r ON ur."roleId" = r.id WHERE r.type="photographer" AND u."countryCode"="UY";',
    "Find all albums for the photographer 'enfocouy'.": 'SELECT * FROM album a WHERE a."ownerId" = (SELECT id FROM user WHERE alias = "enfocouy");',
    "List all surf albums that are not related to an event.": 'SELECT * FROM album a LEFT JOIN activity a2 ON a."activityId" = a2.id WHERE a2.name="SURF" AND a."eventId" IS NULL;',
    "List all the activities and their corresponding billing.": 'SELECT act.name AS deporte, SUM(pur."totalPrice") AS facturacion_total FROM activity act JOIN album alb ON act.id = alb."activityId" JOIN purchase pur ON alb."ownerId" = pur."sellerId" WHERE pur.status = "approved" GROUP BY act.name ORDER BY facturacion_total DESC;',
    "Tell me the photographers with more sales in the event 'Valentin Martinez 2023'.": 'SELECT SUM(p."totalPrice") AS total_ventas FROM purchase p JOIN album a ON p."ownerId" = a."ownerId" JOIN event e ON a."eventId" = e.id WHERE e.name = "Valentín Martínez 2023" AND p.status = "approved";',
    "Which are the users with most purchases since last year?": 'SELECT u.id, u."firstName", u."lastName", COUNT(p.id) AS total_ventas FROM public.user u JOIN purchase p ON u."id" = p."ownerId" WHERE p.status = "approved" AND p."createdAt" > CURRENT_DATE - INTERVAL "1 year" GROUP BY u.id, u."firstName", u."lastName" ORDER BY total_ventas DESC LIMIT 5;',
    "Which is the album with more photographs?": 'SELECT a.id, a.description, COUNT(*) AS photograph_count FROM photograph ph LEFT JOIN album a ON ph."albumId" = a.id GROUP BY a.id, a.description, a."takenDate" ORDER BY photograph_count DESC LIMIT 1;',
    "Find the average number of photographs per album.": 'SELECT AVG(photo_count) FROM (SELECT COUNT(*) AS photo_count FROM photograph GROUP BY "albumId") AS album_photo_counts;',
    "List all albums with no associated purchases.": 'SELECT a.id, a.description FROM album a LEFT JOIN purchase p ON a."ownerId" = p."ownerId" WHERE p.id IS NULL;',
    "List all users who have not made any purchases yet.": 'SELECT * FROM public.user u LEFT JOIN purchase p ON u.id = p."ownerId" WHERE p.id IS NULL;',
    "List all events with the total number of albums created for each.": 'SELECT e.id, e.name, COUNT(a.id) AS total_albums FROM event e LEFT JOIN album a ON e.id = a."eventId" GROUP BY e.id, e.name;',
    "Find the events with the highest total revenue.": 'SELECT e.id, e.name, SUM(p."totalPrice") AS total_revenue FROM event e LEFT JOIN album a ON e.id = a."eventId" LEFT JOIN purchase p ON a."ownerId" = p."sellerId" WHERE p.status = "approved" GROUP BY e.id, e.name ORDER BY total_revenue DESC LIMIT 5;',
}

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)
custom_tool_list = [retriever_tool]

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

prefix: str = """You are an agent designed to interact with a PostgreSQL database.\n
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.\n
Unless the user specifies a specific way of responding, present the results in a table.\n
Unless the user asks for a specific number, or for all of the examples they wish to obtain, always limit your query to at most {top_k} results.\n
You can order the results by a relevant column to return the most interesting examples in the database.\n
Never query for all the columns from a specific table, only ask for the relevant columns given the question.\n
You have access to tools for interacting with the database.\n
Only use the below tools. Only use the information returned by the below tools to construct your final answer.\n
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n\n
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. This is absolutely prohibited.\n\n
If the question does not seem related to the database, just return "I dont know" as the answer.\n
"""

custom_suffix = """
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=prefix,
    top_k=5,
    suffix=custom_suffix,
    extra_tools=custom_tool_list,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
