import openai
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import celtics_engine
from prompts import new_prompt, instruction_str, context

load_dotenv()

os.environ["OPENAI_API_KEY"] = 'put openai API key here'
openai.api_key = os.environ["OPENAI_API_KEY"]

stats_path = os.path.join("data", "FourFactorsV2.csv")
stats_df = pd.read_csv(stats_path)

stats_query_engine = PandasQueryEngine(df=stats_df, verbose=True, instruction_str=instruction_str)
stats_query_engine.update_prompts({"pandas prompt": new_prompt})

tools = [
    QueryEngineTool(
        query_engine=stats_query_engine,
        metadata=ToolMetadata(
            name="nba_stats_data",
            description="this gives information regarding nba team-level statistics for the 1996-97 to 2023-24 seasons",
        ),
    ),
    QueryEngineTool(
        query_engine=celtics_engine,
        metadata=ToolMetadata(
            name="celtics_data",
            description="this gives detailed information about the Boston Celtics playoff games as well as "
                        "information about the regular season for the 2023-24 season",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.chat(prompt)
    print(result)
