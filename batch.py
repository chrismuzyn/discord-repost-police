import discord
from discord.ext import commands
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from queue_producer import publish_message, publish_processed_message
import traceback
import asyncio

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

parser = argparse.ArgumentParser(description='Batch process Discord messages')
parser.add_argument('--channel', type=str, help='Specific channel name to process')
parser.add_argument('--list-channels', action='store_true', help='List all available channels')
args = parser.parse_args()

intents = discord.Intents.all()
disc_client = commands.Bot(intents=intents, command_prefix='/')


async def fetch_messages(channel):
    max_retries = 3
    retry_delay = 5
    message_count = 0
    last_message_id = None
    
    for attempt in range(max_retries):
        try:
            message_count = 0
            last_message_id = None
            
            print(f"[{datetime.now().isoformat()}] STARTING CHANNEL READ - Channel: {channel.name} | Attempt: {attempt + 1}/{max_retries}")
            
            async for message in channel.history(limit=None):
                last_message_id = message.id
                try:
                    timestamp = datetime.now().isoformat()
                    content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
                    print(f"[{timestamp}] READ FROM DISCORD - Channel: {channel.name} | Message ID: {message.id} | Author: {message.author.name}#{message.author.discriminator} (ID: {message.author.id}) | Content: {content_preview} | Created: {message.created_at.isoformat()}")
                    
                    publish_message(message)
                    publish_processed_message(message.id)
                    
                    message_count += 1
                    
                    if message_count % 10 == 0:
                        print(f"[{datetime.now().isoformat()}] PROGRESS - Channel: {channel.name} | Messages processed: {message_count}")
                        
                except discord.errors.HTTPException as e:
                    if e.status == 429:
                        retry_after = e.retry_after
                        print(f"[{datetime.now().isoformat()}] RATE LIMITED - Channel: {channel.name} | Message ID: {message.id} | Retry after: {retry_after}s")
                        await asyncio.sleep(retry_after)
                        publish_message(message)
                        publish_processed_message(message.id)
                        message_count += 1
                    else:
                        print(f"[{datetime.now().isoformat()}] HTTP ERROR - Channel: {channel.name} | Message ID: {message.id} | Status: {e.status} | Error: {str(e)}")
                        raise
                        
                except Exception as e:
                    print(f"[{datetime.now().isoformat()}] ERROR PROCESSING MESSAGE - Channel: {channel.name} | Message ID: {message.id} | Error: {str(e)}")
                    traceback.print_exc()
                    continue
            
            print(f"[{datetime.now().isoformat()}] FINISHED READING CHANNEL - Channel: {channel.name} | Total messages: {message_count}")
            return
            
        except discord.errors.Forbidden:
            print(f"[{datetime.now().isoformat()}] PERMISSION ERROR - Channel: {channel.name} | Error: Bot does not have permission to read this channel")
            return
            
        except discord.errors.NotFound:
            print(f"[{datetime.now().isoformat()}] NOT FOUND ERROR - Channel: {channel.name} | Error: Channel not found or was deleted")
            return
            
        except discord.errors.HTTPException as e:
            print(f"[{datetime.now().isoformat()}] DISCORD HTTP ERROR - Channel: {channel.name} | Status: {e.status} | Error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"[{datetime.now().isoformat()}] RETRYING - Channel: {channel.name} | Waiting {retry_delay}s before attempt {attempt + 2}")
                await asyncio.sleep(retry_delay)
                continue
            else:
                print(f"[{datetime.now().isoformat()}] MAX RETRIES EXCEEDED - Channel: {channel.name}")
                raise
                
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] UNEXPECTED ERROR - Channel: {channel.name} | Last message ID: {last_message_id} | Messages processed: {message_count}")
            print(f"[{datetime.now().isoformat()}] ERROR DETAILS: {str(e)}")
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                print(f"[{datetime.now().isoformat()}] RETRYING - Channel: {channel.name} | Waiting {retry_delay}s before attempt {attempt + 2}")
                await asyncio.sleep(retry_delay)
            else:
                print(f"[{datetime.now().isoformat()}] MAX RETRIES EXCEEDED - Channel: {channel.name}")
                raise


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
