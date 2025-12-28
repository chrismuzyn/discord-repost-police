import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from processor import process_message

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.all()
disc_client = commands.Bot(intents=intents, command_prefix='/')


async def fetch_messages(channel):
    try:
        async for message in channel.history(limit=None):
            await process_message(message, reply=False)
    except:
        print("Couldn't read channel.")


@disc_client.event
async def on_ready():
    print('Logged in as {0.user}'.format(disc_client))
    for guild in disc_client.guilds:
        for channel in guild.text_channels:
            await fetch_messages(channel)


disc_client.run(DISCORD_TOKEN)
