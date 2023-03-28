"""Tool for the selection-inference reasoning."""

from langchain.tools.factor_base import BaseFactorTool
import copy

ACTION_PROMPT = """List possible sequence of actions that can be taken to answer a question:
Example:
Question: <question>
List of action steps:
1. Action or Sequence of actions 1
2. Action or Sequence of actions 2
3. Action or Sequence of actions 3
...

Now list the actions that can be taken to answer the question:
Question: {question}
"""

EVAL_ACTIONS_PROMPT = """Evaluate the actions and return the best one.
Question: <question>
Actions:
"""


class ActionSelection(BaseFactorTool):
    """Tool that allows the model to search through and evaluate actions before taking them."""

    name = "Action Selection Tool"
    description = (
        "A tool useful for searching through and evaluating actions before taking them."
        "Pass the question to the tool and expect an answer for which steps to take next."
        "Eg: Action Input: <question>"
    )

    def _run(self, query: str) -> str:
        """Use the tool."""
        prompt = ACTION_PROMPT.format(question=query)

        # search through the space of actions
        actions =self.llm(prompt, stop=["Question:","Answer:", "Final Answer:", "A:"])
        actions = actions.split("\n")
        actions = [s.split('.')[1].strip() for s in actions]

        # evaluate the actions and return the best one
        eval_prompt = EVAL_ACTIONS_PROMPT.format(question=query)
        for i in range(len(actions)):
            eval_prompt += str(i+1) + ". " + actions[i] + "\nJustification:" 
            justification = self.llm(eval_prompt, stop=["Question:", "A:"+"\n", f"{i+1}.", "Justification:"])
            eval_prompt += justification 
        eval_prompt += "So, best action is:"
        best_action = self.llm(eval_prompt, stop=["Question:", "A:"+"\n", "Justification:"])
        return best_action

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("action selection does not support async")

