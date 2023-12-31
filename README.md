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
Choose one of the quantized models from https://huggingface.co/TheBloke/Llama-2-7B-GGML/tree/main.

### 4. Create the configuration file
Create a file called `config.yaml` in the cloned repository, structured as follows:

```yaml
lists: # Add lists of interest from arxiv, e.g.,
  - "cs.LG"
  - "cs.AI"
  - "cs.CV"
  - "cs.GL"
  - "cs.NE"
bot:
  class_name: # Currently supports SlackBot and TelegramBot
  parameters:
    token: <your token here>
    channel: <your chat id here> # Only for SlackBot
    chat_id: <your chat id here> # Only for TelegramBot
criteria: # Add some keywords to detect topics of interest, e.g.,
  - interpretability
  - xai
  - explainability
model_path: "/path/to/llama-2-7b-chat.ggmlv3.q8_0.bin"
# Choose as summarizer one of the classes defined in `summarizers.py`
summarizer: BulletListSumarizer
```

## See it in action!

Explainable and interpretable AI: https://t.me/+ENLgtQWBzHk2OWU0

Federated Learning and Tiny ML: https://t.me/+onXvUQpsJpUwZTY0

## TBI
- Compatibility with other messaging platforms:
    - [ ] Discord
