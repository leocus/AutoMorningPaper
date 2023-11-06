import pickle
import asyncio
import requests
from bots import BotRegister
from bs4 import BeautifulSoup
from omegaconf import OmegaConf
from summarize_arxiv import summarize

# Load cfg
cfg = OmegaConf.load('./config.yaml')

print(cfg)
bot = BotRegister().make_bot(cfg)

criteria = cfg["criteria"]

try:
    seen: set = pickle.load(open("./seen.pkl", "rb"))
except:
    seen = set()

# Loop through the titles and links and print them
async def main():
    for lst in cfg["lists"]:
        # Parse the HTML content using BeautifulSoup
        url = f"https://arxiv.org/list/{lst}/pastweek?show=496"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all the dt elements which contain the titles and links
        titles = soup.find_all("dt")
        descriptions = soup.find_all("dd")

        for i, title in enumerate(titles):
            href: str = None
            title_text: str = None
            links = title.findAll("a")
            spans = descriptions[i].findAll("div", {"class": "list-title"})
            title_text = spans[0].text.strip()
            for link in links:
                if link.get("href"):
                    href = link.get("href")
                    break

            if href not in seen:
                for kw in criteria:
                    if kw.lower() in title_text.lower():
                        print(f"Processing {title_text}")
                        title_link = "https://arxiv.org" + href
                        print(f"<b>{title_text}</b>\nLink: {title_link}\n")
                        print(href)
                        summary = summarize(href.split("/")[-1])

                        message = f"<b>{title_text}</b>\n\n{summary}\n\nLink: {title_link}"
                        print(message)

                        bot.send_message(text=message)
                        break
                seen.add(href)
    pickle.dump(seen, open("./seen.pkl", "wb"))

if __name__ == '__main__':
    asyncio.run(main())
