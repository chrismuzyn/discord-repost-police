import discord
from discord.ext import commands
import os
import argparse
from dotenv import load_dotenv
from queue_producer import publish_message, publish_processed_message

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

parser = argparse.ArgumentParser(description='Batch process Discord messages')
parser.add_argument('--channel', type=str, help='Specific channel name to process')
parser.add_argument('--list-channels', action='store_true', help='List all available channels')
args = parser.parse_args()

intents = discord.Intents.all()
disc_client = commands.Bot(intents=intents, command_prefix='/')


async def fetch_messages(channel):
    try:
        async for message in channel.history(limit=None):
            await publish_message(message)
            await publish_processed_message(message.id)
    except:
        print("Couldn't read channel.")


async def list_all_channels():
    print("Available channels:")
    for guild in disc_client.guilds:
        print(f"\nServer: {guild.name}")
        for channel in guild.text_channels:
            print(f"  - {channel.name}")
    await disc_client.close()


@disc_client.event
async def on_ready():
    print('Logged in as {0.user}'.format(disc_client))
    
    if args.list_channels:
        await list_all_channels()
        return
    
    for guild in disc_client.guilds:
        for channel in guild.text_channels:
            if args.channel:
                if channel.name == args.channel:
                    print(f"Processing channel: {channel.name}")
                    await fetch_messages(channel)
            else:
                await fetch_messages(channel)


disc_client.run(DISCORD_TOKEN)
