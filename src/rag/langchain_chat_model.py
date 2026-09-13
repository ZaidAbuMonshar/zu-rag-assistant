"""
LangChain ChatModel wrapper around the Qwen2.5-3B-Instruct llama-cpp-python
model (generate.py's load_llm()).

Same incremental pattern as langchain_retriever.py / langchain_hybrid_retriever.py:
adapt an existing, verified piece to a LangChain interface without changing
its behavior. Reuses load_llm() as-is, so n_ctx=16384 and the model path
stay identical to the production /ask path -- no config duplicated here.

Scope note: this wrapper only does "messages in, AIMessage out". It does
NOT include generate.py's context-overflow retry (drop the lowest-scoring
chunk and retry) -- that logic needs to know about *scored chunks*, which
is a prompt-construction concern, not something a ChatModel should know
about. That logic resurfaces when we build the full retrieval->prompt->
model chain next.

Run:
    python langchain_chat_model.py
"""

import sys
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult

from generate import load_llm


class ChatQwenLlamaCpp(BaseChatModel):
    """Wraps a loaded llama_cpp.Llama (Qwen2.5-3B-Instruct) as a LangChain ChatModel."""

    # `Any`, same reasoning as the retriever wrappers: skip pydantic trying
    # to validate/serialize the Llama instance as a model field.
    llm: Any
    temperature: float = 0.2
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "qwen2.5-3b-instruct-llama-cpp"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = convert_to_openai_messages(messages)

        response = self.llm.create_chat_completion(
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        content = response["choices"][0]["message"]["content"].strip()

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    from langchain_core.messages import HumanMessage, SystemMessage

    chat = ChatQwenLlamaCpp(llm=load_llm())
    messages = [
        SystemMessage(content="أنت مساعد يجيب باختصار وبالعربية فقط."),
        HumanMessage(content="ما عاصمة الأردن؟"),
    ]
    result = chat.invoke(messages)
    print(f"Response: {result.content}")
