# flake8: noqa
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
PREFIX_FACTOR = """Answer the following questions as best you can. If you have enough information, answer the question directly.
If not, you have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

FORMAT_INSTRUCTIONS_FACTOR = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do, answer the question directly if you know the answer and say I know the final answer
Action: the action to take, infer the answer directly or should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

SUFFIX_FACTOR = """If you don't find a tool repeatedly, ask a human to teach you the tool. Begin!

Question: {input}
Thought:{agent_scratchpad}"""
