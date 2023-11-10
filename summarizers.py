from abc import abstractmethod
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import TokenTextSplitter
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
    def __init__(self, llm):
        self._llm = llm
        self._chain = None

    def get_chain(self) -> LLMChain:
        if self._chain is None:
            raise ValueError("Invalid summarizer")
        return self._chain

    def summarize(self, text):
        return self.get_chain().run(text)


class PlainTextSummarizer(Summarizer, metaclass=SummarizerRegistry):
    def __init__(self, llm):
        super().__init__(llm)
        self._chain = load_summarize_chain(
            llm=self._llm,
            chain_type='stuff',
        )

class BulletListSumarizer(Summarizer, metaclass=SummarizerRegistry):
    def __init__(self, llm):
        super().__init__(llm)
        template = PromptTemplate(
            input_variables=["paper"],
            template="<|system|>You are a helpful scientific bot. Your job is to make precise summaries of papers|></s><|user|>Here is a paper:\n{paper}\n\nMake a concise summary of the paper using a bullet list.</s><|assistant|>",
        )
        self._chain = LLMChain(llm=self._llm, prompt=template)

class RecursiveSummarizer():
    def __init__(self, summarizer):
        self._summarizer = summarizer

        template = PromptTemplate(
            input_variables=["summary"],
            template="Here is a summary of a paper:\n{summary}\n\nClean this summary by removing repeated entries and unnecessary information.",
        )
        self._refine_chain = LLMChain(llm=self._summarizer._llm, prompt=template)

        # Leaves some margin for the prompt (probably, it can be reduced)
        self._text_splitter = TokenTextSplitter(chunk_size=context_length - 300, chunk_overlap=10)

    def summarize(self, text):
        summary = ""
        for split in self._text_splitter.split_text(text):
            summary += self._summarizer.summarize(split) + "\n"
        return summary
