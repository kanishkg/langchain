"""Tool for the selection-inference reasoning."""

from langchain.tools.factor_base import BaseFactorTool
import copy

L1_PROMPT = """List possible intended meanings of an utterance. Then list the alternate utterances for each meaning.
Finally, choose the best meaning for the utterance based on the alternate utterances.:
Example:
utterance: <question>
1. Meaning 1
Alternate Utterances: <utterance 1> , <utterance 2> , <utterance 3> , ...
2. Meaning 2
Alternate Utterances: <utterance 1> , <utterance 2> , <utterance 3> , ...
3. Meaning 3
Alternate Utterances: <utterance 1> , <utterance 2> , <utterance 3> , ...
...
Best meaning: <meaning>

Now list the meanings for the utterance:
Utterance: {question}
"""

class RSA(BaseFactorTool):
    """Tool that allows the model to search through the possible meanings of an utterance."""

    name = "RSA Tool"
    description = (
        "A tool useful for understanding the possible meanings of a user quesry."
        "Use this before trying to clarify with the user."
        "The tool will return the best meaning for the user query."
        "The tools uses rational speech acts to understand the user query."
        "Use this tool only if the user query is not clear."
        "Eg: Action Input: <user utterance>"
    )

    def _run(self, query: str) -> str:
        """Use the tool."""
        prompt = L1_PROMPT.format(question=query)

        # search through the space of actions
        meanings =self.llm(prompt, stop=["utterance:","Question:", "A:"])
        meanings = meanings.split("\n")
        best_meaning = ""
        for m in meanings:
            if m.startswith("Best meaning:"):
                best_meaning = m.split("Best meaning:")[1].strip()
        return best_meaning

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("RSA does not support async")

