import pika
import json
import os
import base64
import traceback
from datetime import datetime
from dotenv import load_dotenv
from processor import ingest, neuralhash, image_tags, message_tags, embed, hashlib, Image, io, convert_to_png, initialize_database, process_message
from pillow_heif import register_heif_opener

register_heif_opener()

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
QUEUE_NAME = os.getenv("QUEUE_NAME", "discord_messages_queue")
DLQ_NAME = os.getenv("DLQ_NAME", "discord_messages_queue_dlq")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class MockMessage:
    def __init__(self, message_data):
        self.id = message_data['id']
        self.content = message_data['content']
        self.guild = MockGuild(message_data['guild_id'])
        self.channel = MockChannel(message_data['channel_id'])
        self.author = MockAuthor(message_data)
        self.created_at = datetime.fromisoformat(message_data['created_at'])
        self.attachments = [MockAttachment(att) for att in message_data.get('attachments', [])]
        self.client = MockClient(DISCORD_TOKEN)
        self.reference = MockReference(message_data.get('reference_message_id'))
        thread_id = message_data.get('thread_id')
        self.thread = MockThread(thread_id, message_data.get('thread_parent_id')) if thread_id else None

class MockGuild:
    def __init__(self, guild_id):
        self.id = int(guild_id) if guild_id else None

class MockChannel:
    def __init__(self, channel_id):
        self.id = int(channel_id)
        self._messages = {}

    def get_channel(self, channel_id):
        return MockChannel(channel_id)

    async def send(self, content):
        print(f"Mock send to channel {self.id}: {content}")

    async def fetch_message(self, message_id):
        if message_id not in self._messages:
            raise Exception(f"Message {message_id} not found")
        return self._messages[message_id]

class MockAuthor:
    def __init__(self, message_data):
        self.id = int(message_data['author_id']) if message_data['author_id'] is not None else None
        self.name = message_data['author_name']
        self.discriminator = message_data['author_discriminator']
        self.bot = message_data['bot']

class MockReference:
    def __init__(self, message_id):
        self.message_id = int(message_id) if message_id else None

class MockThread:
    def __init__(self, thread_id, parent_id):
        self.id = int(thread_id) if thread_id else None
        self.parent_id = int(parent_id) if parent_id else None

def extract_bot_id_from_token(token):
    parts = token.split('.')
    if len(parts) >= 2:
        try:
            payload = parts[1]
            payload += '=' * ((4 - len(payload) % 4) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            import json
            data = json.loads(decoded)
            return data.get('user_id')
        except:
            pass
    return None

class MockClient:
    def __init__(self, token):
        self.token = token
        self._http_client = None
        bot_id = extract_bot_id_from_token(token)
        if bot_id is None:
            print(f"[{datetime.now().isoformat()}] WARNING - Could not extract bot ID from DISCORD_TOKEN, bot client.user.id will be None")
        self.user = MockAuthor({
            'author_id': bot_id,
            'author_name': 'Bot',
            'author_discriminator': '0000',
            'bot': True
        })

    def get_channel(self, channel_id):
        return MockChannel(channel_id)

    async def close(self):
        if self._http_client:
            await self._http_client.close()

class MockAttachment:
    def __init__(self, attachment_data):
        self.id = attachment_data['id']
        self.filename = attachment_data['filename']
        self.content_type = attachment_data['content_type']
        self.size = attachment_data['size']
        self.url = attachment_data['url']
        self._bytes = attachment_data.get('bytes')

    async def read(self):
        if self._bytes:
            return base64.b64decode(self._bytes.encode('utf-8'))
        else:
            print(f"[{datetime.now().isoformat()}] WARNING - No stored bytes for attachment {self.filename}, using URL as fallback")
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    return await response.read()

async def process_queue_message(message_data):
    reply = message_data.get('reply', False)
    message = MockMessage(message_data)
    await process_message(message, reply)

def on_message(ch, method, properties, body):
    message_data = None
    message_id = None
    
    try:
        import asyncio
        message_data = json.loads(body)
        message_id = message_data.get('id')
        print(f"Processing message {message_id}")
        
        asyncio.run(process_queue_message(message_data))
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Successfully processed message {message_id}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Failed message body: {body[:500]}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except KeyError as e:
        print(f"Missing key error: {e}")
        if message_data:
            print(f"Message data: {json.dumps(message_data, indent=2)[:500]}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except ValueError as e:
        print(f"Value error: {e}")
        print(f"Message ID: {message_id}")
        print(f"Traceback: {traceback.format_exc()}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print(f"Message ID: {message_id}")
        print(f"Traceback: {traceback.format_exc()}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        print(f"Message ID: {message_id or 'unknown'}")
        if message_data:
            print(f"Message data: {json.dumps(message_data, indent=2, default=str)[:500]}")
        print(f"Full traceback: {traceback.format_exc()}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def main():
    initialize_database()
    
    parameters = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(
        queue=DLQ_NAME,
        durable=True
    )

    channel.queue_declare(
        queue=QUEUE_NAME,
        durable=True,
        arguments={
            'x-dead-letter-exchange': '',
            'x-dead-letter-routing-key': DLQ_NAME
        }
    )

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)

    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()

if __name__ == "__main__":
    main()
