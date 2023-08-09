# AutoMorningPaper

Do you fancy your own zero-effort "The morning paper"? Fear not! With the power of LLMs and LangChain you can run your own ArXiv paper summarizer that automatically scans for new papers and sends them to you over Telegram!

## Installation
### 1. Download the repo and install the requirements:
```
git clone https://github.com/leocus/AutoMorningPaper
pip install -r ./requirements.txt
```
### 2. Set up the bot
Create a bot on Telegram using [BotFather](https://t.me/botfather) and get the token.
Then, send a message to the bot and retrieve the chat id from `https://api.telegram.org/bot<TOKEN>/getUpdates`

### 3. Download Llama 2
Choose one of the quantized models from https://huggingface.co/TheBloke/Llama-2-7B-GGML/tree/main and change the model path in `./summarize_arxiv.py`.

### 4. Create the configuration file
Create a file called `config.yaml` in the cloned repository, structured as follows:

```yaml
lists: # Add lists of interest from arxiv, e.g.,
  - "cs.LG"
  - "cs.AI"
  - "cs.CV"
  - "cs.GL"
  - "cs.NE"
token: # Add your token here
chat_id: # Add your chat id here
criteria: # Add some keywords to detect topics of interest, e.g.,
  - interpretability
  - xai
  - explainability
```
