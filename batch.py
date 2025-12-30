import discord
from discord.ext import commands
import os
import argparse
from datetime import datetime
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
        message_count = 0
        async for message in channel.history(limit=None):
            timestamp = datetime.now().isoformat()
            content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
            print(f"[{timestamp}] READ FROM DISCORD - Channel: {channel.name} | Message ID: {message.id} | Author: {message.author.name}#{message.author.discriminator} (ID: {message.author.id}) | Content: {content_preview} | Created: {message.created_at.isoformat()}")
            await publish_message(message)
            await publish_processed_message(message.id)
            message_count += 1
        print(f"[{datetime.now().isoformat()}] FINISHED READING CHANNEL - Channel: {channel.name} | Total messages: {message_count}")
    except:
        print(f"[{datetime.now().isoformat()}] ERROR - Couldn't read channel: {channel.name}")


async def list_all_channels():
    print("Available channels:")
    for guild in disc_client.guilds:
        print(f"\nServer: {guild.name}")
        for channel in guild.text_channels:
            print(f"  - {channel.name}")
    await disc_client.close()


@disc_client.event
async def on_ready():
    print(f"[{datetime.now().isoformat()}] LOGGED IN - Bot: {disc_client.user}")
    
    if args.list_channels:
        await list_all_channels()
        return
    
    for guild in disc_client.guilds:
        print(f"[{datetime.now().isoformat()}] PROCESSING GUILD - Server: {guild.name} (ID: {guild.id}) | Text channels: {len(guild.text_channels)}")
        for channel in guild.text_channels:
            if args.channel:
                if channel.name == args.channel:
                    print(f"[{datetime.now().isoformat()}] PROCESSING CHANNEL - Channel: {channel.name} (ID: {channel.id})")
                    await fetch_messages(channel)
            else:
                print(f"[{datetime.now().isoformat()}] PROCESSING CHANNEL - Channel: {channel.name} (ID: {channel.id})")
                await fetch_messages(channel)


disc_client.run(DISCORD_TOKEN)
