import typer
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import ArxivLoader
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.manager import CallbackManager
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def summarize(arxiv_id):
    text = ArxivLoader(query=arxiv_id, load_max_docs=2).load()
    text = text[0].page_content[:]  # all pages of the Document content

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="/path/to/llama-2-7b-chat.ggmlv3.q8_0.bin",
        n_ctx=1024*8,
        n_parts=-1,
        f16_kv=True,
        logits_all=False,
        vocab_only=False,
        use_mmap=True,
        use_mlock=False,
        n_threads=None,
        n_batch=512,
        temperature=1.0,
        max_tokens=512,
        top_p=0.90,
        top_k=40,
        streaming=True,
        last_n_tokens_size=64,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=50
    )

    docs = text_splitter.create_documents([text])

    num_tokens = llm.get_num_tokens(text)

    print(f"This paper has {num_tokens} tokens in it")

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
    )
    output = summary_chain.run(docs)
    print(output)
    return output


if __name__ == '__main__':
    typer.run(summarize)
