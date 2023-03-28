from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools.selection_inference.tool import SelectionInference
from langchain.tools.human.tool import HumanInputRun
from langchain.tools.interaction.tool import StdInInquireTool
from langchain.tools.llm_tool.tool import LLMTool
from langchain.tools.action_selection.tool import ActionSelection
from langchain.tools.rsa.tool import RSA
from langchain.tools.amplification.tool import Subquestions

llm = OpenAI(model_name="text-davinci-003", temperature=0)
# max depth of 2

# lowest depth
base_tools = [LLMTool(llm=llm), StdInInquireTool(), HumanInputRun(), RSA(llm=llm),
            ActionSelection(llm=llm)]
worker_agent_0 = initialize_agent(base_tools, llm, agent="zero-shot-factor", verbose=True)
worker_agent_0.agent.worker_level = 1

# middle depth
tools_1 = base_tools+[SelectionInference(agent=worker_agent_0, llm=llm), Subquestions(agent=worker_agent_0, llm=llm)]
worker_agent_1 = initialize_agent(tools_1,
                                llm, agent="zero-shot-factor", verbose=True)
worker_agent_1.agent.worker_level = 2

# highest depth
tools = base_tools+[SelectionInference(agent=worker_agent_1, llm=llm), Subquestions(agent=worker_agent_1, llm=llm) ]
agent = initialize_agent(tools, llm, agent="zero-shot-factor", verbose=True)

question = """What is the effect of creatine on cognition?"""

agent.run(question)

