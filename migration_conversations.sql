-- Migration: Add conversations table and conversation_id to attachment_hashes
-- Date: 2025-01-05
-- Description: Adds conversation tracking with semantic coherence detection

-- Create the conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL DEFAULT 'discord',
    channel_id VARCHAR(255) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    message_count INT NOT NULL DEFAULT 1,
    representative_embedding vector(4096)
);

-- Add conversation_id foreign key column to attachment_hashes
ALTER TABLE attachment_hashes
ADD COLUMN IF NOT EXISTS conversation_id BIGINT REFERENCES conversations(id);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_attachment_hashes_conversation_id ON attachment_hashes (conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_channel_id_last_message_at ON conversations (channel_id, last_message_at);

-- Optional: Backfill existing messages with NULL conversation_id (they will get assigned on next processing)
-- No action needed - existing messages will have NULL conversation_id which is valid
