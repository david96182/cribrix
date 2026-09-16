-- Runs once, on first initialisation of the data volume.
-- Idempotent so it stays safe if the volume is ever re-seeded.
CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram index support, for optional hybrid (lexical + vector) retrieval.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
