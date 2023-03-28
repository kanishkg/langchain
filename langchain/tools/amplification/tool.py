"""Tool for the selection-inference reasoning."""

from langchain.tools.factor_base import BaseFactorTool
from langchain.utilities.bing_search import BingSearchAPIWrapper
import copy

SUBQUESTION_PROMPT = """Split the question into subquestions. The subquestions could depend on answers to previous subquestions.
For example:
question: "What is the effect of creatine on cognition?"
subquestions:
Q: What is creatine?
Q: What is cognition?
Q: How does creatine affect cognition?
Q: What are the benefits of creatine on cognition?
Q: What are the side effects of creatine on cognition?

question: "What is the square of Will Smith's age?"
Q: What is Will Smith's age?
Q: What is the square of Will Smith's age?

question:{question}
subquestions:
Q:"""


class Subquestions(BaseFactorTool):
    """Tool that allows the model to ask multiple subquestions and answer them."""

    name = "Subquestions Tool"
    description = (
        "A tool useful for breaking a question down into simpler subquestions. "
        "When asked a complex question, use this tool to break it down into simpler subquestions."
        "Pass the question to the tool and expect the answer from the subquestions."
        "Eg: Action Input: <question>"
    )

    def _run(self, query: str) -> str:
        """Use the tool."""
        prompt = SUBQUESTION_PROMPT.format(question=query)
        subquestions ="Q:"+self.llm(prompt, stop=["question:","Answer:", "Final Answer:", "A:"])
        # split subquestions into list
        subquestions = subquestions.split("\n")
        subquestions = [s.split(':')[1].strip() for s in subquestions]

        ## TODO: run subquestions in parallel? They need to be independent of each other
        reasoning_trace = ""
        for i in range(len(subquestions)):
            reasoning_trace += "Q: " + subquestions[i]
            answer = self.agent.run(subquestions[i])
            reasoning_trace += "A: " + answer + "\n"
        reasoning_trace += "Original Question: " + query
        final_answer = self.agent.run(reasoning_trace)
        return final_answer

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("subquestions does not support async")

