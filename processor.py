import discord
import hashlib
import psycopg2
import random
import onnxruntime
import numpy
import io
import base64
import json
from PIL import Image
from pillow_heif import register_heif_opener
from urllib.parse import urlparse
from text_generators.insult_generator import hit_me
from dotenv import load_dotenv
import os
from openai import OpenAI
from datetime import datetime

register_heif_opener()

def parse_vector_string(vector_str):
    import numpy as np
    if vector_str is None:
        return None
    if isinstance(vector_str, list):
        return np.array(vector_str, dtype=np.float32)
    if isinstance(vector_str, np.ndarray):
        return vector_str
    try:
        parsed = json.loads(vector_str.replace('[', '').replace(']', '').replace(' ', ','))
        return np.array(parsed, dtype=np.float32)
    except (json.JSONDecodeError, AttributeError):
        try:
            return np.array([float(x) for x in str(vector_str).strip('[]').split(',')], dtype=np.float32)
        except ValueError:
            return None

def retry_on_failure(max_retries, exceptions):
    import functools
    import time
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    print(f"processor.py:retry [{datetime.now().isoformat()}] - Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                    time.sleep(2 ** attempt)
            return None
        return wrapper
    return decorator

load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
print(f"processor.py:15 [{datetime.now().isoformat()}] - Loaded .env file")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOSTNAME = os.getenv("DB_HOSTNAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_EMBEDDING_BASE_URL = os.getenv("OPENAI_EMBEDDING_BASE_URL")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", 600))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", 0))
CONVERSATION_TIMEOUT = int(os.getenv("CONVERSATION_TIMEOUT", 1800))
SEMANTIC_DISTANCE_THRESHOLD = float(os.getenv("SEMANTIC_DISTANCE_THRESHOLD", 0.7))
SOURCE_IDENTIFIER = os.getenv("SOURCE_IDENTIFIER", "discord")

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key="not-needed",
    timeout=OPENAI_TIMEOUT
)

embedding_client = OpenAI(
    base_url=OPENAI_EMBEDDING_BASE_URL,
    api_key="not-needed",
    timeout=OPENAI_TIMEOUT
)

print(f"processor.py:27 [{datetime.now().isoformat()}] - Connecting to DB at {DB_HOSTNAME}")
db_conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOSTNAME, port=5432)
db_cursor = db_conn.cursor()
print(f"processor.py:30 [{datetime.now().isoformat()}] - Connected to DB successfully")

def initialize_database():
    print(f"processor.py:initialize_database [{datetime.now().isoformat()}] - Initializing database")
    try:
        print(f"processor.py:initialize_database [{datetime.now().isoformat()}] - Using existing DB connection at {DB_HOSTNAME}")
        
        print(f"processor.py:initialize_database [{datetime.now().isoformat()}] - Creating/verifying table structure")
        db_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        db_cursor.execute("""
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
        db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                source VARCHAR(50) NOT NULL DEFAULT 'discord',
                channel_id VARCHAR(255) NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_message_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                message_count INT NOT NULL DEFAULT 1,
                representative_embedding vector(4096)
            );
        """)
        db_cursor.execute("""
            ALTER TABLE attachment_hashes
            ADD COLUMN IF NOT EXISTS conversation_id BIGINT REFERENCES conversations(id);
        """)
        db_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attachment_hashes_conversation_id ON attachment_hashes (conversation_id);
        """)
        db_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_channel_id_last_message_at ON conversations (channel_id, last_message_at);
        """)
        db_conn.commit()
        print(f"processor.py:initialize_database [{datetime.now().isoformat()}] - Table structure verified and committed")
    except Exception as e:
        print(f"processor.py:initialize_database - FATAL ERROR: Database initialization failed: {e}")
        import sys
        sys.exit(1)

print(f"processor.py:47 [{datetime.now().isoformat()}] - Loading ONNX model")
sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 2
session = onnxruntime.InferenceSession(
    "apple-neuralhash/model.onnx",
    sess_options=sess_options
)
print(f"processor.py:48 [{datetime.now().isoformat()}] - ONNX model loaded")

print(f"processor.py:49 [{datetime.now().isoformat()}] - Loading seed file")
seed1 = open("apple-neuralhash/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = numpy.frombuffer(seed1, dtype=numpy.float32)
seed1 = seed1.reshape([96, 128])
print(f"processor.py:52 [{datetime.now().isoformat()}] - Seed file loaded and reshaped")

"""
Convert unsupported images, return None if conversion not needed.
"""
def convert_to_png(attachment_bytes, filename=None):
    print(f"processor.py:77 [{datetime.now().isoformat()}] - convert_to_png: Starting - filename={filename}")
    try:
        image = Image.open(io.BytesIO(attachment_bytes))
        image_format = image.format or 'UNKNOWN'
        print(f"processor.py:80 [{datetime.now().isoformat()}] - convert_to_png: Detected format={image_format}")
        
        if image_format in ['WEBP', 'HEIF', 'HEIC']:
            print(f"processor.py:82 [{datetime.now().isoformat()}] - convert_to_png: Converting {image_format} to PNG")
            image = image.convert('RGB')
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            png_bytes = buffered.getvalue()
            print(f"processor.py:86 [{datetime.now().isoformat()}] - convert_to_png: Conversion complete")
            return png_bytes
        else:
            print(f"processor.py:88 [{datetime.now().isoformat()}] - convert_to_png: No conversion needed for {image_format}")
            return None
    except Exception as e:
        print(f"processor.py:90 [{datetime.now().isoformat()}] - convert_to_png: Error - {e}")
        return None

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


async def ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word, reply=False, tags=None, vector=None, orig_text=None, message_data=None):
    print(f"processor.py:69 [{datetime.now().isoformat()}] - ingest: Starting - message_id={message_id}, md5={md5_hash[:8]}..., visual={visual_hash}")
    
    if tags is None or not tags:
        print(f"processor.py:70 [{datetime.now().isoformat()}] - ingest: ERROR - tags is None or empty: {tags}")
        raise Exception(f"Tags cannot be None or empty. tags={tags}")
    
    print(f"processor.py:71 [{datetime.now().isoformat()}] - ingest: Tags validated, count={len(tags)}")
    
    if message_data is None:
        message_data = {}
    
    conversation_id = detect_conversation(message_data, vector, channel_id, message_date)
    print(f"processor.py:74 [{datetime.now().isoformat()}] - ingest: conversation_id={conversation_id}")
    
    #print(f"processor.py:75 [{datetime.now().isoformat()}] - ingest: Querying DB for existing hashes")
    #db_cursor.execute('SELECT DISTINCT ON (md5_hash, visual_hash) message_id, channel_id FROM attachment_hashes WHERE (server_id = %s) AND (md5_hash = %s OR (visual_hash = %s AND visual_hash != \'l\')) ORDER BY md5_hash, visual_hash, message_date ASC', (server_id, md5_hash, visual_hash))
    #existing_message = db_cursor.fetchone()
    #print(f"processor.py:77 [{datetime.now().isoformat()}] - ingest: DB query complete, existing_message={existing_message is not None}")
    
    #if existing_message != None:
    #    print(f"processor.py:74 [{datetime.now().isoformat()}] - ingest: Found existing match in channel {existing_message[1]}")
    #    off_channel = message.client.get_channel(int(existing_message[1]))
    #    try:
    #        print(f"processor.py:76 [{datetime.now().isoformat()}] - ingest: Fetching Discord message {existing_message[0]}")
    #        off_message = await off_channel.fetch_message(existing_message[0])
    #        print(f"processor.py:78 [{datetime.now().isoformat()}] - ingest: Discord message fetched successfully")
    #    except Exception as e:
    #        print(f"processor.py:80 [{datetime.now().isoformat()}] - ingest: ERROR fetching Discord message: {e}")
    #        print("Inserting image that we can't find anymore.")
    #        if reply:
    #            print(f"processor.py:83 [{datetime.now().isoformat()}] - ingest: Sending Discord reply about missing message")
    #            await message.channel.send("👮 I have this file/link already but I can't find the message it came from.  I'll let you off this time.")
    #        print(f"processor.py:89 [{datetime.now().isoformat()}] - ingest: Deleting old entry from DB")
    #        db_cursor.execute('DELETE FROM attachment_hashes WHERE message_id = %s AND channel_id = %s', (existing_message[0], existing_message[1]))
    #        db_conn.commit()
    #        print(f"processor.py:92 [{datetime.now().isoformat()}] - ingest: Inserting new entry to DB")
    #        db_cursor.execute('INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text, conversation_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (md5_hash, visual_hash, server_id, str(channel_id), message_id, message_date, json.dumps(tags), vector, orig_text, conversation_id))
    #        db_conn.commit()
    #        update_conversation(conversation_id, vector)
    #        print(f"processor.py:95 [{datetime.now().isoformat()}] - ingest: Completed (missing message case)")
    #        return
    #    
    #    print("Inserting image that we already have a match for.")
    #    print(f"processor.py:98 [{datetime.now().isoformat()}] - ingest: Inserting duplicate entry to DB")
    #    db_cursor.execute('INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text, conversation_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (md5_hash, visual_hash, server_id, str(channel_id), message_id, message_date, json.dumps(tags), vector, orig_text, conversation_id))
    #    db_conn.commit()
    #    update_conversation(conversation_id, vector)
    #    print(f"processor.py:101 [{datetime.now().isoformat()}] - ingest: DB insert complete")

    #    if reply:
    #        print(f"processor.py:100 [{datetime.now().isoformat()}] - ingest: Preparing Discord reply")
    #        original_msg_url = off_message.jump_url
    #        insult = hit_me()
    #        if visual_hash != 'l':
    #            possible_responses = [
    #                "Here's the original post that was probably also from reddit, you {0}.".format(insult),
    #                "Do you both browse reddit together, you {0}.".format(insult),
    #                "You {0}, do you even read this chat?".format(insult),
    #                "Ya, I'm gonna have to bring you down to the station.",
    #                "Fucking {0}.".format(insult) ]
    #            response = random.choice(possible_responses)
    #        else:
    #            possible_responses = ["You {0}, do you even read this chat?".format(insult), "Ya, I'm gonna have to bring you down to the station.", "Fucking {0}.".format(insult)]
    #            if "reddit.com" in word:
    #                response = "Is reddit down for anybody else?"
    #            else:
    #                response = random.choice(possible_responses)

    #        print(f"processor.py:117 [{datetime.now().isoformat()}] - ingest: Sending Discord reply")
    #        await message.reply("🚨🚨🚨\n{0}\n{1}".format(response,original_msg_url))
    #        print(f"processor.py:119 [{datetime.now().isoformat()}] - ingest: Discord reply sent")

    #else:
    print("Inserting new image.")
    print(f"processor.py:125 [{datetime.now().isoformat()}] - ingest: Inserting new entry to DB")
    db_cursor.execute('INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date, tags, vector, orig_text, conversation_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (md5_hash, visual_hash, server_id, str(channel_id), message_id, message_date, json.dumps(tags), vector, orig_text, conversation_id))
    db_conn.commit()
    update_conversation(conversation_id, vector)
    print(f"processor.py:128 [{datetime.now().isoformat()}] - ingest: Completed (new image case)")

def image_tags(attachment, filename=None):
    print(f"processor.py:127 [{datetime.now().isoformat()}] - image_tags: Starting")
    converted_png = convert_to_png(attachment, filename)
    bytes_to_process = converted_png if converted_png else attachment
    
    image = Image.open(io.BytesIO(bytes_to_process)).convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()

    print(f"processor.py:134 [{datetime.now().isoformat()}] - image_tags: Calling OpenAI API for image tagging")
    try:
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
                                "url": f"data:image/png;base64,{base64.b64encode(img_str).decode()}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=16000
        )
        print(f"processor.py:151 [{datetime.now().isoformat()}] - image_tags: OpenAI API response received")
        print(f"processor.py:151 [{datetime.now().isoformat()}] - image_tags: Full API response: {response.model_dump_json()}")
    except Exception as e:
        print(f"processor.py:152 [{datetime.now().isoformat()}] - image_tags: ERROR calling OpenAI API: {e}")
        raise Exception(f"Failed to get tags from OpenAI API: {e}")

    response_content = response.choices[0].message.content.strip()
    print(f"processor.py:153 [{datetime.now().isoformat()}] - image_tags: Raw API response: '{response_content}'")

    if not response_content:
        print(f"processor.py:154 [{datetime.now().isoformat()}] - image_tags: ERROR - Empty response from API")
        raise Exception("OpenAI API returned empty response for image tags")

    tags = None

    try:
        import ast
        tags = ast.literal_eval(response_content)
        if isinstance(tags, list):
            print(f"processor.py:155 [{datetime.now().isoformat()}] - image_tags: Parsed as JSON array")
            tags = [str(tag).strip() for tag in tags]
    except (SyntaxError, ValueError):
        print(f"processor.py:156 [{datetime.now().isoformat()}] - image_tags: Not a JSON array, parsing as comma-separated")
        tags = [tag.strip() for tag in response_content.split(',')]

    tags = [tag for tag in tags if tag and tag.strip()]
    print(f"processor.py:157 [{datetime.now().isoformat()}] - image_tags: Parsed {len(tags)} tags: {tags}")

    if not tags:
        print(f"processor.py:158 [{datetime.now().isoformat()}] - image_tags: ERROR - No tags parsed from response")
        raise Exception(f"No valid tags found in API response: '{response_content}'")

    return tags

def message_tags(message):
    print(f"processor.py:157 [{datetime.now().isoformat()}] - message_tags: Starting")
    print(f"processor.py:159 [{datetime.now().isoformat()}] - message_tags: Calling OpenAI API for message tagging")
    try:
        response = client.chat.completions.create(
            model="GLM-4.6V",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate around 12-15 tags that describe the content and meaning of this Discord message. Return only the tags as a comma-separated list, nothing else.\n\nMessage: {message.content}"
                }
            ],
            max_tokens=16000
        )
        print(f"processor.py:170 [{datetime.now().isoformat()}] - message_tags: OpenAI API response received")
        print(f"processor.py:170 [{datetime.now().isoformat()}] - message_tags: Full API response: {response.model_dump_json()}")
    except Exception as e:
        print(f"processor.py:171 [{datetime.now().isoformat()}] - message_tags: ERROR calling OpenAI API: {e}")
        raise Exception(f"Failed to get tags from OpenAI API: {e}")

    response_content = response.choices[0].message.content.strip()
    print(f"processor.py:172 [{datetime.now().isoformat()}] - message_tags: Raw API response: '{response_content}'")

    if not response_content:
        print(f"processor.py:173 [{datetime.now().isoformat()}] - message_tags: ERROR - Empty response from API")
        raise Exception("OpenAI API returned empty response for message tags")

    tags = None

    try:
        import ast
        tags = ast.literal_eval(response_content)
        if isinstance(tags, list):
            print(f"processor.py:174 [{datetime.now().isoformat()}] - message_tags: Parsed as JSON array")
            tags = [str(tag).strip() for tag in tags]
    except (SyntaxError, ValueError):
        print(f"processor.py:175 [{datetime.now().isoformat()}] - message_tags: Not a JSON array, parsing as comma-separated")
        tags = [tag.strip() for tag in response_content.split(',')]

    tags = [tag for tag in tags if tag and tag.strip()]
    print(f"processor.py:176 [{datetime.now().isoformat()}] - message_tags: Parsed {len(tags)} tags: {tags}")

    if not tags:
        print(f"processor.py:177 [{datetime.now().isoformat()}] - message_tags: ERROR - No tags parsed from response")
        raise Exception(f"No valid tags found in API response: '{response_content}'")

    return tags

def embed(text, image_bytes=None):
    print(f"processor.py:175 [{datetime.now().isoformat()}] - embed: Starting")
    
    if image_bytes is not None:
        print(f"processor.py:176 [{datetime.now().isoformat()}] - embed: Processing multimodal embedding (image + text)")
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text or ""
                    }
                ]
            }
        ]
        
        print(f"processor.py:177 [{datetime.now().isoformat()}] - embed: Calling OpenAI API for multimodal embedding")
        response = embedding_client.embeddings.create(
            model="Qwen3-VL-Embedding",
            messages=messages
        )
    else:
        print(f"processor.py:177 [{datetime.now().isoformat()}] - embed: Processing text-only embedding")
        print(f"processor.py:177 [{datetime.now().isoformat()}] - embed: Calling OpenAI API for embedding")
        response = embedding_client.embeddings.create(
            model="Qwen3-VL-Embedding",
            input=text
        )
    
    print(f"processor.py:182 [{datetime.now().isoformat()}] - embed: OpenAI API response received")
    print(f"processor.py:183 [{datetime.now().isoformat()}] - embed: Completed")
    return response.data[0].embedding

def calculate_cosine_similarity(embedding1, embedding2):
    print(f"processor.py:342 [{datetime.now().isoformat()}] - calculate_cosine_similarity: Starting")
    import numpy as np
    vec1 = parse_vector_string(embedding1)
    vec2 = parse_vector_string(embedding2)
    if vec1 is None or vec2 is None:
        print(f"processor.py:345 [{datetime.now().isoformat()}] - calculate_cosine_similarity: ERROR - Failed to parse vectors")
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        print(f"processor.py:347 [{datetime.now().isoformat()}] - calculate_cosine_similarity: ERROR - Zero norm vector")
        return 0.0
    similarity = dot_product / (norm1 * norm2)
    print(f"processor.py:348 [{datetime.now().isoformat()}] - calculate_cosine_similarity: Completed - similarity={similarity:.4f}")
    return similarity

def get_latest_message_in_channel(channel_id):
    print(f"processor.py:350 [{datetime.now().isoformat()}] - get_latest_message_in_channel: Starting - channel_id={channel_id}")
    db_cursor.execute('''
        SELECT id, message_id, channel_id, message_date, conversation_id, vector
        FROM attachment_hashes
        WHERE channel_id = %s
        ORDER BY message_date DESC
        LIMIT 1
    ''', (str(channel_id),))
    result = db_cursor.fetchone()
    if result and result[5]:
        result = list(result)
        result[5] = parse_vector_string(result[5])
        result = tuple(result)
    print(f"processor.py:358 [{datetime.now().isoformat()}] - get_latest_message_in_channel: Completed - found={result is not None}")
    return result

def get_conversation_representative_embedding(conversation_id):
    print(f"processor.py:360 [{datetime.now().isoformat()}] - get_conversation_representative_embedding: Starting - conversation_id={conversation_id}")
    db_cursor.execute('''
        SELECT representative_embedding
        FROM conversations
        WHERE id = %s
    ''', (conversation_id,))
    result = db_cursor.fetchone()
    if result and result[0]:
        parsed_embedding = parse_vector_string(result[0])
        print(f"processor.py:367 [{datetime.now().isoformat()}] - get_conversation_representative_embedding: Completed - found embedding")
        return parsed_embedding
    print(f"processor.py:368 [{datetime.now().isoformat()}] - get_conversation_representative_embedding: Completed - no embedding found")
    return None

def create_conversation(channel_id, initial_embedding):
    print(f"processor.py:370 [{datetime.now().isoformat()}] - create_conversation: Starting - channel_id={channel_id}")
    db_cursor.execute('''
        INSERT INTO conversations (source, channel_id, started_at, last_message_at, message_count, representative_embedding)
        VALUES (%s, %s, now(), now(), 1, %s)
        RETURNING id
    ''', (SOURCE_IDENTIFIER, str(channel_id), initial_embedding))
    conversation_id = db_cursor.fetchone()[0]
    db_conn.commit()
    print(f"processor.py:378 [{datetime.now().isoformat()}] - create_conversation: Completed - conversation_id={conversation_id}")
    return conversation_id

def update_conversation(conversation_id, new_embedding):
    print(f"processor.py:380 [{datetime.now().isoformat()}] - update_conversation: Starting - conversation_id={conversation_id}")
    db_cursor.execute('''
        SELECT message_count, representative_embedding
        FROM conversations
        WHERE id = %s
    ''', (conversation_id,))
    result = db_cursor.fetchone()
    if not result:
        print(f"processor.py:387 [{datetime.now().isoformat()}] - update_conversation: ERROR - conversation not found")
        return
    
    current_count = result[0]
    current_avg_embedding = parse_vector_string(result[1])
    
    import numpy as np
    new_embedding = parse_vector_string(new_embedding)
    if current_avg_embedding is None or new_embedding is None:
        print(f"processor.py:393 [{datetime.now().isoformat()}] - update_conversation: ERROR - Failed to parse embeddings")
        return
    new_avg_embedding = ((current_avg_embedding * current_count) + new_embedding) / (current_count + 1)
    
    db_cursor.execute('''
        UPDATE conversations
        SET last_message_at = now(),
            message_count = message_count + 1,
            representative_embedding = %s
        WHERE id = %s
    ''', (new_avg_embedding.tolist(), conversation_id))
    db_conn.commit()
    print(f"processor.py:402 [{datetime.now().isoformat()}] - update_conversation: Completed - conversation_id={conversation_id}, new_count={current_count + 1}")

def get_conversation_id_by_message_id(message_id):
    print(f"processor.py:404 [{datetime.now().isoformat()}] - get_conversation_id_by_message_id: Starting - message_id={message_id}")
    db_cursor.execute('''
        SELECT conversation_id
        FROM attachment_hashes
        WHERE message_id = %s
        LIMIT 1
    ''', (message_id,))
    result = db_cursor.fetchone()
    if result and result[0]:
        print(f"processor.py:411 [{datetime.now().isoformat()}] - get_conversation_id_by_message_id: Completed - conversation_id={result[0]}")
        return result[0]
    print(f"processor.py:412 [{datetime.now().isoformat()}] - get_conversation_id_by_message_id: Completed - no conversation_id found")
    return None

def detect_conversation(message_data, embedding, channel_id, message_date):
    print(f"processor.py:414 [{datetime.now().isoformat()}] - detect_conversation: Starting - channel_id={channel_id}")
    
    reference_message_id = message_data.get('reference_message_id')
    thread_id = message_data.get('thread_id')
    
    print(f"processor.py:418 [{datetime.now().isoformat()}] - detect_conversation: Layer 1 - Checking explicit signals")
    print(f"processor.py:419 [{datetime.now().isoformat()}] - detect_conversation: reference_message_id={reference_message_id}, thread_id={thread_id}")
    
    if reference_message_id:
        print(f"processor.py:421 [{datetime.now().isoformat()}] - detect_conversation: Message is a reply")
        parent_conversation_id = get_conversation_id_by_message_id(reference_message_id)
        if parent_conversation_id:
            print(f"processor.py:423 [{datetime.now().isoformat()}] - detect_conversation: Found parent conversation_id={parent_conversation_id}")
            return parent_conversation_id
        else:
            print(f"processor.py:425 [{datetime.now().isoformat()}] - detect_conversation: Parent message not found in DB, creating new conversation")
            return create_conversation(channel_id, embedding)
    
    if thread_id:
        print(f"processor.py:429 [{datetime.now().isoformat()}] - detect_conversation: Message is in a thread")
        db_cursor.execute('''
            SELECT conversation_id
            FROM attachment_hashes
            WHERE message_id = %s OR message_id = %s
            LIMIT 1
        ''', (thread_id, message_data.get('thread_parent_id')))
        thread_result = db_cursor.fetchone()
        if thread_result and thread_result[0]:
            print(f"processor.py:438 [{datetime.now().isoformat()}] - detect_conversation: Found thread conversation_id={thread_result[0]}")
            return thread_result[0]
        else:
            print(f"processor.py:440 [{datetime.now().isoformat()}] - detect_conversation: Thread not found in DB, creating new conversation")
            return create_conversation(channel_id, embedding)
    
    print(f"processor.py:444 [{datetime.now().isoformat()}] - detect_conversation: Layer 2 - Checking heuristic signals")
    latest_message = get_latest_message_in_channel(channel_id)
    
    if not latest_message:
        print(f"processor.py:447 [{datetime.now().isoformat()}] - detect_conversation: No previous message found, creating new conversation")
        return create_conversation(channel_id, embedding)
    
    latest_message_date = latest_message[3]
    message_date = message_date.replace(tzinfo=None)
    time_diff = (message_date - latest_message_date).total_seconds()
    print(f"processor.py:451 [{datetime.now().isoformat()}] - detect_conversation: time_diff={time_diff}s, timeout={CONVERSATION_TIMEOUT}s")
    
    if time_diff > CONVERSATION_TIMEOUT:
        print(f"processor.py:453 [{datetime.now().isoformat()}] - detect_conversation: Time diff exceeds timeout, creating new conversation")
        return create_conversation(channel_id, embedding)
    
    print(f"processor.py:456 [{datetime.now().isoformat()}] - detect_conversation: Layer 3 - Checking semantic coherence")
    candidate_conversation_id = latest_message[4]
    
    if not candidate_conversation_id:
        print(f"processor.py:459 [{datetime.now().isoformat()}] - detect_conversation: Latest message has no conversation_id, creating new conversation")
        return create_conversation(channel_id, embedding)
    
    conversation_embedding = get_conversation_representative_embedding(candidate_conversation_id)
    
    if conversation_embedding is None:
        print(f"processor.py:464 [{datetime.now().isoformat()}] - detect_conversation: No conversation embedding found, creating new conversation")
        return create_conversation(channel_id, embedding)
    
    similarity = calculate_cosine_similarity(embedding, conversation_embedding)
    print(f"processor.py:467 [{datetime.now().isoformat()}] - detect_conversation: similarity={similarity:.4f}, threshold={SEMANTIC_DISTANCE_THRESHOLD}")
    
    if similarity >= SEMANTIC_DISTANCE_THRESHOLD:
        print(f"processor.py:469 [{datetime.now().isoformat()}] - detect_conversation: Message is on-topic, using existing conversation_id={candidate_conversation_id}")
        return candidate_conversation_id
    else:
        print(f"processor.py:471 [{datetime.now().isoformat()}] - detect_conversation: Message is off-topic, creating new conversation")
        return create_conversation(channel_id, embedding)

async def _process_content(message, reply, content_type, content_data=None):
    print(f"processor.py:186 [{datetime.now().isoformat()}] - _process_content: Starting - content_type={content_type}")
    
    if content_type == 'url':
        word = content_data
        print(f"processor.py:189 [{datetime.now().isoformat()}] - _process_content: Processing URL: {word}")
        md5_hash = hashlib.md5(word.lower().encode()).hexdigest()
        visual_hash = 'l'
        print(f"processor.py:192 [{datetime.now().isoformat()}] - _process_content: Calling message_tags for URL")
        tags = message_tags(message)
        word_for_check = word
        image_bytes_for_embedding = None
    
    elif content_type == 'attachment':
        attachment = content_data
        print(f"processor.py:196 [{datetime.now().isoformat()}] - _process_content: Processing attachment: {attachment.filename}")
        attachment_bytes = await attachment.read()
        try:
            converted_png = convert_to_png(attachment_bytes, attachment.filename)
            bytes_for_image = converted_png if converted_png else attachment_bytes
            image = Image.open(io.BytesIO(bytes_for_image)).convert('RGB')
            print(f"processor.py:202 [{datetime.now().isoformat()}] - _process_content: Image opened successfully")
        except Exception as e:
            print(f"processor.py:204 [{datetime.now().isoformat()}] - _process_content: Failed to open image: {e}")
            return None
        
        md5_hash = hashlib.md5(attachment_bytes).hexdigest()
        print(f"processor.py:208 [{datetime.now().isoformat()}] - _process_content: Calling neuralhash for attachment")
        visual_hash = neuralhash(image)
        print(f"processor.py:209 [{datetime.now().isoformat()}] - _process_content: Calling image_tags for attachment")
        tags = image_tags(attachment_bytes, attachment.filename)
        word_for_check = ""
        image_bytes_for_embedding = bytes_for_image
    
    elif content_type == 'text':
        print(f"processor.py:212 [{datetime.now().isoformat()}] - _process_content: Processing text-only message")
        md5_hash = hashlib.md5(message.content.encode()).hexdigest()
        visual_hash = 't'
        print(f"processor.py:215 [{datetime.now().isoformat()}] - _process_content: Calling message_tags for text-only message")
        #tags = message_tags(message)
        tags = [""]
        word_for_check = ""
        image_bytes_for_embedding = None
    
    else:
        print(f"processor.py:219 [{datetime.now().isoformat()}] - _process_content: ERROR - Unknown content_type: {content_type}")
        return None
    
    server_id = message.guild.id
    channel_id = message.channel.id
    message_id = message.id
    message_date = message.created_at
    
    print(f"processor.py:225 [{datetime.now().isoformat()}] - _process_content: Calling embed for content")
    vector = embed(message.content, image_bytes_for_embedding)
    orig_text = message.content
    
    message_data = {
        'reference_message_id': str(message.reference.message_id) if message.reference and message.reference.message_id else None,
        'thread_id': str(message.thread.id) if message.thread else None,
        'thread_parent_id': str(message.thread.parent_id) if message.thread and message.thread.parent_id else None
    }
    
    print(f"processor.py:233 [{datetime.now().isoformat()}] - _process_content: Calling ingest for {content_type}")
    await ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word_for_check, reply, tags, vector, orig_text, message_data)
    print(f"processor.py:234 [{datetime.now().isoformat()}] - _process_content: Completed")
    return True

async def process_message(message, reply=False):
    print(f"processor.py:186 [{datetime.now().isoformat()}] - process_message: Starting - message_id={message.id}, reply={reply}")
    if message.author == message.client.user or message.author.bot:
        print(f"processor.py:189 [{datetime.now().isoformat()}] - process_message: Skipping bot message")
        return

    print(f"processor.py:192 [{datetime.now().isoformat()}] - process_message: Checking for URLs in message content")
    for word in message.content.split():
        parsed = urlparse(word.lower())
        if parsed.hostname and len(parsed.path) > 4 and "discord.com" not in parsed.hostname:
            await _process_content(message, reply, 'url', word)

    print(f"processor.py:216 [{datetime.now().isoformat()}] - process_message: Checking for attachments")
    if len(message.attachments) > 0:
        print(f"processor.py:218 [{datetime.now().isoformat()}] - process_message: Found {len(message.attachments)} attachment(s)")
        for attachment in message.attachments:
            await _process_content(message, reply, 'attachment', attachment)

    print(f"processor.py:253 [{datetime.now().isoformat()}] - process_message: Checking for text-only messages")
    has_urls = any(urlparse(word.lower()).hostname and len(urlparse(word.lower()).path) > 4 and "discord.com" not in urlparse(word.lower()).hostname for word in message.content.split())
    
    print(f"MESSAGE CONTENT STRIP: {message.content.strip()}")
    print(f"HASURLS: {has_urls}")
    if len(message.attachments) == 0 and not has_urls and message.content.strip():
        await _process_content(message, reply, 'text')
    
    print(f"processor.py:249 [{datetime.now().isoformat()}] - process_message: Completed")
