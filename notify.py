import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from queue_producer import publish_message

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.all()
disc_client = commands.Bot(intents=intents, command_prefix='/')


@disc_client.event
async def on_ready():
    print('Logged in as {0.user}'.format(disc_client))


@disc_client.event
async def on_message(message):
    if message.author == disc_client.user:
        return
    if message.author.bot:
        return

    await publish_message(message)


disc_client.run(DISCORD_TOKEN)
