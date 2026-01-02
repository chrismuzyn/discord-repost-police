import discord
import hashlib
import psycopg2
import random
import onnxruntime
import numpy
import io
from PIL import Image
from urllib.parse import urlparse
from text_generators.insult_generator import hit_me
from dotenv import load_dotenv
import os
from openai import OpenAI
from datetime import datetime

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
print(f"processor.py:15 [{datetime.now().isoformat()}] - Loaded .env file")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOSTNAME = os.getenv("DB_HOSTNAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key="not-needed"
)

print(f"processor.py:27 [{datetime.now().isoformat()}] - Connecting to DB at {DB_HOSTNAME}")
db_conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOSTNAME, port=5432)
db_cursor = db_conn.cursor()
print(f"processor.py:30 [{datetime.now().isoformat()}] - Connected to DB successfully")

print(f"processor.py:31 [{datetime.now().isoformat()}] - Creating/verifying table structure")
with db_conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attachment_hashes (
            id SERIAL PRIMARY KEY,
            server_id BIGINT NOT NULL,
            channel_id VARCHAR(20) NOT NULL,
            message_id VARCHAR(20) NOT NULL,
            md5_hash VARCHAR(32) NOT NULL,
            visual_hash VARCHAR(32) NOT NULL,
            message_date TIMESTAMP NOT NULL,
            tags JSONB,
            vector vector(4096),
            orig_text TEXT
        );
    """)
print(f"processor.py:46 [{datetime.now().isoformat()}] - Table structure verified")

print(f"processor.py:47 [{datetime.now().isoformat()}] - Loading ONNX model")
session = onnxruntime.InferenceSession("apple-neuralhash/model.onnx")
print(f"processor.py:48 [{datetime.now().isoformat()}] - ONNX model loaded")

print(f"processor.py:49 [{datetime.now().isoformat()}] - Loading seed file")
seed1 = open("apple-neuralhash/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = numpy.frombuffer(seed1, dtype=numpy.float32)
seed1 = seed1.reshape([96, 128])
print(f"processor.py:52 [{datetime.now().isoformat()}] - Seed file loaded and reshaped")


def neuralhash(image):
    print(f"processor.py:54 [{datetime.now().isoformat()}] - neuralhash: Starting")
    image = image.resize([360, 360])
    arr = numpy.array(image).astype(numpy.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

    inputs = {session.get_inputs()[0].name: arr}
    print(f"processor.py:60 [{datetime.now().isoformat()}] - neuralhash: Running inference")
    outs = session.run(None, inputs)
    
    print(f"processor.py:63 [{datetime.now().isoformat()}] - neuralhash: Computing hash")
    hash_output = seed1.dot(outs[0].flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
    hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
    print(f"processor.py:66 [{datetime.now().isoformat()}] - neuralhash: Completed")
    return hash_hex


async def check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word, reply=False, tags=None, vector=None, orig_text=None):
    print(f"processor.py:69 [{datetime.now().isoformat()}] - check_and_ingest: Starting - message_id={message_id}, md5={md5_hash[:8]}..., visual={visual_hash}")
    print(f"processor.py:70 [{datetime.now().isoformat()}] - check_and_ingest: Querying DB for existing hashes")
    db_cursor.execute('SELECT DISTINCT ON (md5_hash, visual_hash) message_id, channel_id FROM attachment_hashes WHERE (server_id = %s) AND (md5_hash = %s OR (visual_hash = %s AND visual_hash != \'l\')) ORDER BY md5_hash, visual_hash, message_date ASC', (server_id, md5_hash, visual_hash))
    existing_message = db_cursor.fetchone()
    print(f"processor.py:72 [{datetime.now().isoformat()}] - check_and_ingest: DB query complete, existing_message={existing_message is not None}")
    
    if existing_message != None:
        print(f"processor.py:74 [{datetime.now().isoformat()}] - check_and_ingest: Found existing match in channel {existing_message[1]}")
        off_channel = message.client.get_channel(int(existing_message[1]))
        try:
            print(f"processor.py:76 [{datetime.now().isoformat()}] - check_and_ingest: Fetching Discord message {existing_message[0]}")
            off_message = await off_channel.fetch_message(existing_message[0])
            print(f"processor.py:78 [{datetime.now().isoformat()}] - check_and_ingest: Discord message fetched successfully")
        except Exception as e:
            print(f"processor.py:80 [{datetime.now().isoformat()}] - check_and_ingest: ERROR fetching Discord message: {e}")
            print("Inserting image that we can't find anymore.")
            if reply:
                print(f"processor.py:83 [{datetime.now().isoformat()}] - check_and_ingest: Sending Discord reply about missing message")
                await message.channel.send("👮 I have this file/link already but I can't find the message it came from.  I'll let you off this time.")
            print(f"processor.py:86 [{datetime.now().isoformat()}] - check_and_ingest: Deleting old entry from DB")
            db_cursor.execute('DELETE FROM attachment_hashes WHERE message_id = %s AND channel_id = %s', (existing_message[0], existing_message[1]))
            db_conn.commit()
            print(f"processor.py:89 [{datetime.now().isoformat()}] - check_and_ingest: Inserting new entry to DB")
            db_cursor.execute('INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text))
            db_conn.commit()
            print(f"processor.py:92 [{datetime.now().isoformat()}] - check_and_ingest: Completed (missing message case)")
            return
        
        print("Inserting image that we already have a match for.")
        print(f"processor.py:95 [{datetime.now().isoformat()}] - check_and_ingest: Inserting duplicate entry to DB")
        db_cursor.execute('INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text))
        db_conn.commit()
        print(f"processor.py:98 [{datetime.now().isoformat()}] - check_and_ingest: DB insert complete")

        if reply:
            print(f"processor.py:100 [{datetime.now().isoformat()}] - check_and_ingest: Preparing Discord reply")
            original_msg_url = off_message.jump_url
            insult = hit_me()
            if visual_hash != 'l':
                possible_responses = [
                    "Here's the original post that was probably also from reddit, you {0}.".format(insult),
                    "Do you both browse reddit together, you {0}.".format(insult),
                    "You {0}, do you even read this chat?".format(insult),
                    "Ya, I'm gonna have to bring you down to the station.",
                    "Fucking {0}.".format(insult) ]
                response = random.choice(possible_responses)
            else:
                possible_responses = ["You {0}, do you even read this chat?".format(insult), "Ya, I'm gonna have to bring you down to the station.", "Fucking {0}.".format(insult)]
                if "reddit.com" in word:
                    response = "Is reddit down for anybody else?"
                else:
                    response = random.choice(possible_responses)

            print(f"processor.py:117 [{datetime.now().isoformat()}] - check_and_ingest: Sending Discord reply")
            await message.reply("🚨🚨🚨\n{0}\n{1}".format(response,original_msg_url))
            print(f"processor.py:119 [{datetime.now().isoformat()}] - check_and_ingest: Discord reply sent")

    else:
        print("Inserting new image.")
        print(f"processor.py:122 [{datetime.now().isoformat()}] - check_and_ingest: Inserting new entry to DB")
        db_cursor.execute('INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text))
        db_conn.commit()
        print(f"processor.py:125 [{datetime.now().isoformat()}] - check_and_ingest: Completed (new image case)")

def image_tags(attachment):
    print(f"processor.py:127 [{datetime.now().isoformat()}] - image_tags: Starting")
    image = Image.open(io.BytesIO(attachment)).convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()

    print(f"processor.py:134 [{datetime.now().isoformat()}] - image_tags: Calling OpenAI API for image tagging")
    response = client.chat.completions.create(
        model="GLM-4.6V",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate around 12-15 tags that describe this image. Return only the tags as a comma-separated list, nothing else."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str.hex()}"
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )
    print(f"processor.py:151 [{datetime.now().isoformat()}] - image_tags: OpenAI API response received")

    tags_str = response.choices[0].message.content.strip()
    tags = [tag.strip() for tag in tags_str.split(',')]
    print(f"processor.py:154 [{datetime.now().isoformat()}] - image_tags: Completed with {len(tags)} tags")
    return tags

def message_tags(message):
    print(f"processor.py:157 [{datetime.now().isoformat()}] - message_tags: Starting")
    print(f"processor.py:159 [{datetime.now().isoformat()}] - message_tags: Calling OpenAI API for message tagging")
    response = client.chat.completions.create(
        model="GLM-4.6V",
        messages=[
            {
                "role": "user",
                "content": f"Generate around 12-15 tags that describe the content and meaning of this Discord message. Return only the tags as a comma-separated list, nothing else.\n\nMessage: {message.content}"
            }
        ],
        max_tokens=200
    )
    print(f"processor.py:170 [{datetime.now().isoformat()}] - message_tags: OpenAI API response received")

    tags_str = response.choices[0].message.content.strip()
    tags = [tag.strip() for tag in tags_str.split(',')]
    print(f"processor.py:173 [{datetime.now().isoformat()}] - message_tags: Completed with {len(tags)} tags")
    return tags

def embed(text):
    print(f"processor.py:175 [{datetime.now().isoformat()}] - embed: Starting")
    print(f"processor.py:177 [{datetime.now().isoformat()}] - embed: Calling OpenAI API for embedding")
    response = client.embeddings.create(
        model="llama-embed-nemotron",
        input=text
    )
    print(f"processor.py:182 [{datetime.now().isoformat()}] - embed: OpenAI API response received")
    print(f"processor.py:183 [{datetime.now().isoformat()}] - embed: Completed")
    return response.data[0].embedding

async def process_message(message, reply=False):
    print(f"processor.py:186 [{datetime.now().isoformat()}] - process_message: Starting - message_id={message.id}, reply={reply}")
    if message.author == message.client.user or message.author.bot:
        print(f"processor.py:189 [{datetime.now().isoformat()}] - process_message: Skipping bot message")
        pass

    print(f"processor.py:192 [{datetime.now().isoformat()}] - process_message: Checking for URLs in message content")
    for word in message.content.split():
        if urlparse(word.lower()).hostname:
            if len(urlparse(word.lower()).path) > 4:
                if "discord.com" not in urlparse(word.lower()).hostname:
                    print(f"processor.py:197 [{datetime.now().isoformat()}] - process_message: Found URL to process: {word}")
                    md5_hash = hashlib.md5(word.lower().encode()).hexdigest()
                    visual_hash = 'l'
                    server_id = message.guild.id
                    channel_id = message.channel.id
                    message_id = message.id
                    message_date = message.created_at

                    print(f"processor.py:205 [{datetime.now().isoformat()}] - process_message: Calling message_tags for URL")
                    tags = message_tags(message)
                    print(f"processor.py:207 [{datetime.now().isoformat()}] - process_message: Calling embed for URL")
                    vector = embed(message.content)
                    orig_text = message.content

                    print(f"processor.py:211 [{datetime.now().isoformat()}] - process_message: Calling check_and_ingest for URL")
                    await check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word, reply, tags, vector, orig_text)
                    print(f"processor.py:213 [{datetime.now().isoformat()}] - process_message: check_and_ingest completed for URL")

    print(f"processor.py:216 [{datetime.now().isoformat()}] - process_message: Checking for attachments")
    if len(message.attachments) > 0:
        print(f"processor.py:218 [{datetime.now().isoformat()}] - process_message: Found {len(message.attachments)} attachment(s)")
        for attachment in message.attachments:
            print(f"processor.py:220 [{datetime.now().isoformat()}] - process_message: Reading attachment {attachment.filename}")
            attachment_bytes = await attachment.read()
            try:
                image = Image.open(io.BytesIO(attachment_bytes)).convert('RGB')
                print(f"processor.py:225 [{datetime.now().isoformat()}] - process_message: Image opened successfully")
            except Exception as e:
                print(f"processor.py:227 [{datetime.now().isoformat()}] - process_message: Failed to open image: {e}")
                continue

            md5_hash = hashlib.md5(attachment_bytes).hexdigest()
            print(f"processor.py:231 [{datetime.now().isoformat()}] - process_message: Calling neuralhash for attachment")
            visual_hash = neuralhash(image)
            server_id = message.guild.id
            channel_id = message.channel.id
            message_id = message.id
            message_date = message.created_at

            print(f"processor.py:238 [{datetime.now().isoformat()}] - process_message: Calling image_tags for attachment")
            tags = image_tags(attachment_bytes)
            print(f"processor.py:240 [{datetime.now().isoformat()}] - process_message: Calling embed for attachment")
            vector = embed(message.content)
            orig_text = message.content

            print(f"processor.py:244 [{datetime.now().isoformat()}] - process_message: Calling check_and_ingest for attachment")
            await check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, "", reply, tags, vector, orig_text)
            print(f"processor.py:246 [{datetime.now().isoformat()}] - process_message: check_and_ingest completed for attachment")
    
    print(f"processor.py:249 [{datetime.now().isoformat()}] - process_message: Completed")
