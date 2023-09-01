from abc import abstractmethod
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain


class SummarizerRegistry(type):
    REG = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REG[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get(cls, class_name):
        return cls.REG[class_name]


class Summarizer:
    def __init__(self, model_path):
        self._llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4000,
            n_parts=-1,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,
            n_threads=None,
            n_batch=512,
            temperature=0.75,
            max_tokens=1024,
            top_p=0.90,
            top_k=40,
            streaming=True,
            last_n_tokens_size=64,
        )
        self._chain = None

    def get_chain(self) -> LLMChain:
        if self._chain is None:
            raise ValueError("Invalid summarizer")
        return self._chain

    def summarize(self, text):
        return self.get_chain().run(text)


class PlainTextSummarizer(Summarizer, metaclass=SummarizerRegistry):
    def __init__(self, model_path):
        super().__init__(model_path)
        self._chain = load_summarize_chain(
            llm=self._llm,
            chain_type='stuff',
        )

class BulletListSumarizer(Summarizer, metaclass=SummarizerRegistry):
    def __init__(self, model_path):
        super().__init__(model_path)
        template = PromptTemplate(
            input_variables=["paper"],
            template="Here is a paper:\n{paper}\n\nMake a concise summary of the paper using a bullet list.",
        )
        self._chain = LLMChain(llm=self._llm, prompt=template)
