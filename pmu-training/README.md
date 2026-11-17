# PMU France — ingestion and training

This repository contains an isolated Railway service for inspecting and importing the public French horse-racing dataset before model training.

## Data policy
- Dataset source: `annaelmoussa/horse-racing-france` on Hugging Face.
- Raw Parquet files are downloaded at runtime to `/data`; they are never committed to Git.
- The service first inventories dataset configurations, Parquet files and schemas. Model training must only start after the schema and label fields are validated.
- No LONACI arrival labels are mixed into French training labels. LONACI is reserved for a separate external evaluation set.

## Railway
Set `DATA_DIR=/data`. Deploy the service and call `POST /inspect` to enumerate remote Parquet files and schemas. The service exposes `/health` and `/status`.

The first deployment intentionally performs inspection only. It does not silently start a large download or create model metrics from unverified fields.