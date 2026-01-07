import pika
import json
import os
import base64
import asyncio
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
QUEUE_NAME = os.getenv("QUEUE_NAME", "discord_messages_queue")
PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed_discord_messages_queue")
MAX_ATTACHMENT_SIZE_MB = int(os.getenv("MAX_ATTACHMENT_SIZE_MB", "10"))

def get_connection():
    return pika.URLParameters(RABBITMQ_URL)

async def download_attachment_bytes(attachment):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    print(f"[{datetime.now().isoformat()}] ERROR DOWNLOADING ATTACHMENT - URL: {attachment.url} | Status: {response.status}")
                    return None
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] EXCEPTION DOWNLOADING ATTACHMENT - URL: {attachment.url} | Error: {e}")
        return None

async def publish_message(message):
    try:
        connection = pika.BlockingConnection(get_connection())
        channel = connection.channel()

        channel.queue_declare(
            queue=QUEUE_NAME,
            durable=True,
            arguments={
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': f'{QUEUE_NAME}_dlq'
            }
        )

        message_data = {
            'id': str(message.id),
            'content': message.content,
            'author_id': str(message.author.id),
            'author_name': message.author.name,
            'author_discriminator': message.author.discriminator,
            'bot': message.author.bot,
            'guild_id': str(message.guild.id) if message.guild else None,
            'channel_id': str(message.channel.id),
            'created_at': message.created_at.isoformat(),
            'attachments': [],
            'reference_message_id': str(message.reference.message_id) if message.reference and message.reference.message_id else None,
            'thread_id': str(message.thread.id) if message.thread else None,
            'thread_parent_id': str(message.thread.parent_id) if message.thread and message.thread.parent_id else None
        }

        for attachment in message.attachments:
            attachment_data = {
                'id': str(attachment.id),
                'filename': attachment.filename,
                'content_type': attachment.content_type,
                'size': attachment.size,
                'url': attachment.url
            }

            if attachment.size and attachment.size <= MAX_ATTACHMENT_SIZE_MB * 1024 * 1024:
                attachment_bytes = await download_attachment_bytes(attachment)
                if attachment_bytes:
                    attachment_data['bytes'] = base64.b64encode(attachment_bytes).decode('utf-8')
                    print(f"[{datetime.now().isoformat()}] DOWNLOADED ATTACHMENT - Filename: {attachment.filename} | Size: {len(attachment_bytes)} bytes | Message ID: {message_data['id']}")
                else:
                    print(f"[{datetime.now().isoformat()}] ATTACHMENT DOWNLOAD FAILED - Filename: {attachment.filename} | Message ID: {message_data['id']}")
            else:
                print(f"[{datetime.now().isoformat()}] ATTACHMENT SKIPPED (TOO LARGE) - Filename: {attachment.filename} | Size: {attachment.size} bytes | Max: {MAX_ATTACHMENT_SIZE_MB}MB | Message ID: {message_data['id']}")

            message_data['attachments'].append(attachment_data)

        channel.basic_publish(
            exchange='',
            routing_key=QUEUE_NAME,
            body=json.dumps(message_data),
            properties=pika.BasicProperties(
                delivery_mode=2
            )
        )

        print(f"[{datetime.now().isoformat()}] SENT TO QUEUE - Queue: {QUEUE_NAME} | Message ID: {message_data['id']} | Author: {message_data['author_name']}#{message_data['author_discriminator']} | Status: SUCCESS")
        connection.close()
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERROR SENDING TO QUEUE - Queue: {QUEUE_NAME} | Message ID: {message.id if message else 'unknown'} | Error: {e}")
        raise

async def publish_processed_message(message_id):
    try:
        connection = pika.BlockingConnection(get_connection())
        channel = connection.channel()

        channel.queue_declare(
            queue=PROCESSED_QUEUE_NAME,
            durable=True
        )

        processed_data = {
            'message_id': str(message_id)
        }

        channel.basic_publish(
            exchange='',
            routing_key=PROCESSED_QUEUE_NAME,
            body=json.dumps(processed_data),
            properties=pika.BasicProperties(
                delivery_mode=2
            )
        )

        print(f"[{datetime.now().isoformat()}] SENT TO QUEUE - Queue: {PROCESSED_QUEUE_NAME} | Message ID: {processed_data['message_id']} | Status: SUCCESS")
        connection.close()
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ERROR SENDING TO QUEUE - Queue: {PROCESSED_QUEUE_NAME} | Message ID: {message_id} | Error: {e}")
        raise
