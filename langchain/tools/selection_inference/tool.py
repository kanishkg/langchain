"""Tool for the selection-inference reasoning."""

from langchain.tools.factor_base import BaseFactorTool
from langchain.utilities.bing_search import BingSearchAPIWrapper

SELCTION_PROMPT = """
{context}
Return the relevant information from the above context to answer the query below:
{question}
Relevant information:"""

INFERENCE_PROMPT = """
{selection}
Based on the above information, answer the query below:
{question}"""

class SelectionInference(BaseFactorTool):
    """Tool that adds the capability to reason with selection and inference."""

    name = "Selection Inference Tool"
    description = (
        "A tool useful for reasoning with selection and inference. "
        "When asked a question with a lot of information, you can use this tool to find the answer. "
        "The input is passed to the selection module, which selects the relevant information."
        "The inference module then uses the selected information to infer the answer."
        "Pass the full context and the question as a semi-colon separated string."
        "Eg: Action Input: <full long context here>; <question>."
    )

    def _run(self, query: str) -> str:
        """Use the tool."""
        context, question = query.strip().split(";")
        prompt = SELCTION_PROMPT.format(context=context, question=question)
        selection = self.llm(prompt, stop=["Answer:", "Final Answer:", "A:"])
        inference_prompt = INFERENCE_PROMPT.format(selection=selection, question=question)
        inference = self.agent.run(inference_prompt)
        return inference

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SelectionInference does not support async")

