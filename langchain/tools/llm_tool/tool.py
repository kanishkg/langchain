"""Tool for the selection-inference reasoning."""

from langchain.tools.factor_base import BaseFactorTool
from langchain.utilities.bing_search import BingSearchAPIWrapper

QUERY_PROMPT = """
Answer the following question:
Question: {question}
Answer:"""


class LLMTool(BaseFactorTool):
    """Tool that adds the capability to reason with selection and inference."""

    name = "LLM Tool"
    description = (
        "A tool useful for reasoning with an LLM. Use this tool for easy last step inferences."
        "Pass the question as a string."
        "Use it for things like deducing or infering the answer to a question."
        "Eg: Action Input: question"
    )

    def _run(self, query: str) -> str:
        """Use the tool."""
        prompt = QUERY_PROMPT.format(question=query)
        answer = self.llm(prompt, stop=["Question:", "Q:", "A:"])
        return answer

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SelectionInference does not support async")

