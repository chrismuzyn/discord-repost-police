import pika
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
QUEUE_NAME = os.getenv("QUEUE_NAME", "discord_messages_queue")
PROCESSED_QUEUE_NAME = os.getenv("PROCESSED_QUEUE_NAME", "processed_discord_messages_queue")

def get_connection():
    return pika.URLParameters(RABBITMQ_URL)

def publish_message(message):
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

def publish_processed_message(message_id):
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
