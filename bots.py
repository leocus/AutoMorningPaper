"""
File: bots.py
Author: Leonardo Lucio Custode
Github: github.com/leocus
Description: This file contains an implementation of intefaces
            for messaging platforms
"""
import abc
import slack
import telegram


class BotRegister:
    bots = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.bots[cls.__name__] = cls

    def make_bot(cls, kwargs):
        return cls.bots[kwargs.bot.class_name](**kwargs.bot.parameters)


class Bot:
    """This is the base class for messaging bots."""

    @abc.abstractmethod
    async def send_message(self, text: str):
        """
        Sends a message to a pre-determined recipient.

        :text: The text to be sent.
        :returns: None
        """
        pass


class TelegramBot(Bot, BotRegister):
    """A bot interface that interacts with the Telegram APIs."""

    def __init__(self, token: str, chat_id: str):
        """
        Initializes the bot.

        :token: The bot's token returned by BotFather
        :chat_id: The chat where to send the messages
        """
        Bot.__init__(self)

        self._token = token
        self._chat_id = chat_id
        self._bot = telegram.Bot(self._token)

    async def send_message(self, text: str):
        async with self._bot:
            await self._bot.send_message(text=text, chat_id=self._chat_id, parse_mode="HTML")


class SlackBot(Bot, BotRegister):
    """A bot interface that interacts with the Slack APIs."""

    def __init__(self, token: str, channel: str):
        """
        Initializes the bot.

        :token: The bot's token
        :chat_id: The chat where to send the messages
        """
        Bot.__init__(self)

        self._token = token
        self._channel = channel
        self._bot = slack.WebClient(token=token)

    async def send_message(self, text: str):
        self._bot.chat_postMessage(channel=self._channel, text=text)

