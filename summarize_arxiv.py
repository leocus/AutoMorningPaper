import yaml
import typer
from summarizers import SummarizerRegistry
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import LLMChain


cfg = yaml.load(open("./config.yaml"), Loader=yaml.CLoader)
summarizer = SummarizerRegistry.get(cfg["summarizer"])(cfg["model_path"])


def summarize(arxiv_id):
    text = ArxivLoader(query=arxiv_id, load_max_docs=2).load()
    text = text[0].page_content[:]  # all pages of the Document content

    output = summarizer.summarize(text)
    print(output)
    return output


if __name__ == '__main__':
    typer.run(summarize)
