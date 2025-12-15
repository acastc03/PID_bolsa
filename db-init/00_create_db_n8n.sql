-- Create dedicated n8n database if it does not exist
-- This script runs only when the Postgres data directory is initialized.
-- Safe for re-runs using psql's \gexec to conditionally create.

SELECT 'CREATE DATABASE n8n'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'n8n')\gexec;
