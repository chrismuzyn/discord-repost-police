import pika
import json
import os
import aiohttp
import traceback
from datetime import datetime
from dotenv import load_dotenv
from processor import check_and_ingest, neuralhash, image_tags, message_tags, embed, hashlib, Image, io, convert_to_png
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
        self.thread = MockThread(message_data.get('thread_id'), message_data.get('thread_parent_id'))

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
        self.id = int(message_data['author_id'])
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

class MockClient:
    def __init__(self, token):
        self.token = token
        self._http_client = None

    async def get_channel(self, channel_id):
        async with aiohttp.ClientSession() as session:
            session.headers.update({'Authorization': f'Bot {self.token}'})
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

    async def read(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                return await response.read()

async def process_queue_message(message_data):
    reply = message_data.get('reply', False)
    message = MockMessage(message_data)

    for word in message.content.split():
        from urllib.parse import urlparse
        if urlparse(word.lower()).hostname:
            if len(urlparse(word.lower()).path) > 4:
                if "discord.com" not in urlparse(word.lower()).hostname:
                    md5_hash = hashlib.md5(word.lower().encode()).hexdigest()
                    visual_hash = 'l'
                    server_id = message.guild.id
                    channel_id = message.channel.id
                    message_id = message.id
                    message_date = message.created_at

                    tags = message_tags(message)
                    vector = embed(message.content)
                    orig_text = message.content

                    await check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word, reply, tags, vector, orig_text, message_data)

    if len(message.attachments) > 0:
        for attachment in message.attachments:
            attachment_bytes = await attachment.read()
            try:
                converted_png = convert_to_png(attachment_bytes, attachment.filename)
                bytes_for_image = converted_png if converted_png else attachment_bytes
                image = Image.open(io.BytesIO(bytes_for_image)).convert('RGB')
            except:
                continue

            md5_hash = hashlib.md5(attachment_bytes).hexdigest()
            visual_hash = neuralhash(image)
            server_id = message.guild.id
            channel_id = message.channel.id
            message_id = message.id
            message_date = message.created_at

            tags = image_tags(attachment_bytes, attachment.filename)
            vector = embed(message.content)
            orig_text = message.content

            await check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, "", reply, tags, vector, orig_text, message_data)

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
