
from abc import abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Extra, Field, validator

from langchain.tools.base import BaseTool

class BaseFactorTool(BaseTool):
    """Class responsible for defining a tool or skill for an LLM where the LLM itself is a tool."""
    name: str
    description: str   
    agent: object
    llm: object

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.agent.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Factored tools do not support async") 