import yaml
import typer
from langchain.llms import CTransformers
from langchain.document_loaders import ArxivLoader
from summarizers import SummarizerRegistry, RecursiveSummarizer


cfg = yaml.load(open("./config.yaml"), Loader=yaml.CLoader)
llm = CTransformers(model="TheBloke/zephyr-7B-beta-GGUF", model_file="zephyr-7b-beta.Q4_K_M.gguf", model_type="mistral", config={'gpu_layers': cfg['gpu_layers'], 'context_length' : cfg['context_length']})
summarizer = RecursiveSummarizer(SummarizerRegistry.get(cfg["summarizer"])(llm, cfg['context_length']))


def summarize(arxiv_id):
    text = ArxivLoader(query=arxiv_id, load_max_docs=2).load()
    text = text[0].page_content[:]  # all pages of the Document content

    output = summarizer.summarize(text)
    print(output)
    return output


if __name__ == '__main__':
    typer.run(summarize)
