import discord
import os
import hashlib
import asyncio
import base64
import psycopg2
import random
import datetime
import onnxruntime
import numpy
import io
from PIL import Image
from urllib.parse import urlparse
from text_generators.insult_generator import hit_me
from dotenv import load_dotenv

#load .env
load_dotenv(os.path.join(os.path.abspath(os.path.dirname(__file__)), '.env'))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOSTNAME = os.getenv("DB_HOSTNAME")

#set intents
intents = discord.Intents.all()
disc_client = discord.Client(intents=intents)

# Set up etcd client and constants
db_conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOSTNAME, port=5432)
db_cursor = db_conn.cursor()

# create table if it does not exist
with db_conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attachment_hashes (
            id SERIAL PRIMARY KEY,
            server_id BIGINT NOT NULL,
            channel_id VARCHAR(20) NOT NULL,
            message_id VARCHAR(20) NOT NULL,
            md5_hash VARCHAR(32) NOT NULL,
            visual_hash VARCHAR(32) NOT NULL,
            message_date TIMESTAMP NOT NULL
        );
    """)

#load onnx model into ram
session = onnxruntime.InferenceSession("apple-neuralhash/model.onnx")
seed1 = open("apple-neuralhash/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = numpy.frombuffer(seed1, dtype=numpy.float32)
seed1 = seed1.reshape([96, 128])


def neuralhash(attachment):
    image = Image.open(io.BytesIO(attachment)).convert('RGB')
    image = image.resize([360, 360])
    arr = numpy.array(image).astype(numpy.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

    #run model
    inputs = {session.get_inputs()[0].name: arr}
    outs = session.run(None, inputs)
    
    hash_output = seed1.dot(outs[0].flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
    hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
    return hash_hex

#given these parameters, check db for it and ingest
async def check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word):
    db_cursor.execute('SELECT DISTINCT ON (md5_hash, visual_hash) message_id, channel_id FROM attachment_hashes WHERE (server_id = %s) AND (md5_hash = %s OR (visual_hash = %s AND visual_hash != \'l\')) ORDER BY md5_hash, visual_hash, message_date ASC', (server_id, md5_hash, visual_hash))
    existing_message = db_cursor.fetchone()
    # Check if the file has already been uploaded
    if existing_message != None:
        off_channel = disc_client.get_channel(int(existing_message[1]))
        try:
            off_message = await off_channel.fetch_message(existing_message[0])
        except Exception as e:
            #the message probably got deleted
            await message.channel.send("👮 I have this file/link already but I can't find the message it came from.  I'll let you off this time.")
            db_cursor.execute('DELETE FROM attachment_hashes WHERE message_id = %s AND channel_id = %s', (existing_message[0], existing_message[1]))
            db_conn.commit()
            db_cursor.execute(
                #'INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id) VALUES (%s, %s, %s, %s, %s)',
                #(md5_hash, visual_hash, server_id, channel_id, message_id)
                'INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date) VALUES (%s, %s, %s, %s, %s, %s)',
                (md5_hash, visual_hash, server_id, channel_id, message_id, message_date)
            )
            db_conn.commit()
            return
        
        #this is awful
        db_cursor.execute(
            'INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date) VALUES (%s, %s, %s, %s, %s, %s)',
            (md5_hash, visual_hash, server_id, channel_id, message_id, message_date)
        )

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

        await message.reply("🚨🚨🚨\n{0}\n{1}".format(response,original_msg_url))

    else:
        db_cursor.execute(
            #'INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id) VALUES (%s, %s, %s, %s, %s)',
            #(md5_hash, visual_hash, server_id, channel_id, message_id)
            'INSERT INTO attachment_hashes (md5_hash, visual_hash, server_id, channel_id, message_id, message_date) VALUES (%s, %s, %s, %s, %s, %s)',
            (md5_hash, visual_hash, server_id, channel_id, message_id, message_date)
        )
        db_conn.commit()

@disc_client.event
async def on_ready():
    print('Logged in as {0.user}'.format(disc_client))

@disc_client.event
async def on_message(message):
    if message.author == disc_client.user:
        return # Ignore messages from the bot itself
    if message.author.bot:
        return # Ignore messages from bots

    #check message for links
    for word in message.content.split():
        if urlparse(word.lower()).hostname:
            #we are dealing with a live link, not to be confused with a live leak
            #do not bother with links that do not have significant paths
            if len(urlparse(word.lower()).path) > 4:
                if "discord.com" not in urlparse(word.lower()).hostname:
                    md5_hash = hashlib.md5(word.lower().encode()).hexdigest()
                    visual_hash = 'l'
                    server_id = message.guild.id
                    channel_id = message.channel.id
                    message_id = message.id
                    message_date = message.created_at

                    await check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, word)

    if len(message.attachments) > 0:
        for attachment in message.attachments:
            md5_hash = hashlib.md5(await attachment.read()).hexdigest()
            #if attachment is photo
            visual_hash = neuralhash(await attachment.read())
            server_id = message.guild.id
            channel_id = message.channel.id
            message_id = message.id
            message_date = message.created_at

            await check_and_ingest(md5_hash, visual_hash, server_id, channel_id, message_id, message_date, message, "")

disc_client.run(DISCORD_TOKEN)

