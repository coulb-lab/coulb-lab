import json
import os
import threading
from datetime import datetime, timezone

import boto3
import requests
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi, hf_hub_url

REPO = os.getenv("HF_DATASET", "annaelmoussa/horse-racing-france")
IMPORT_TOKEN = os.getenv("IMPORT_TOKEN", "")
app = FastAPI(title="PMU France data preparation", version="0.2.0")
lock = threading.Lock()
state = {
    "phase": "idle", "started_at": None, "finished_at": None,
    "repo": REPO, "files": [], "imported": [], "error": None
}

def now():
    return datetime.now(timezone.utc).isoformat()

def inspect_dataset():
    info = HfApi().dataset_info(REPO, files_metadata=True)
    files = [
        {"path": x.rfilename, "size": getattr(x, "size", None)}
        for x in (info.siblings or []) if x.rfilename.endswith(".parquet")
    ]
    configs = []
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(REPO)
    except Exception as exc:
        configs = ["config_discovery_error: " + str(exc)[:250]]
    return files, configs

@app.get("/health")
def health():
    return {"ok": True, "service": "pmu-data-training", "time": now()}

@app.get("/status")
def status():
    return state

@app.get("/inspect")
@app.post("/inspect")
def inspect():
    try:
        files, configs = inspect_dataset()
        with lock:
            state.update({
                "phase": "inspected", "finished_at": now(), "repo": REPO,
                "configs": configs, "files": files, "parquet_count": len(files),
                "total_bytes": sum(x["size"] or 0 for x in files), "error": None
            })
        return state
    except Exception as exc:
        with lock:
            state.update({"phase": "error", "error": str(exc), "finished_at": now()})
        raise HTTPException(status_code=502, detail=str(exc))

def do_import():
    try:
        with lock:
            state.update({"phase": "importing", "started_at": now(), "finished_at": None,
                          "imported": [], "error": None})
        files, configs = inspect_dataset()
        required = ["S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise RuntimeError("Missing storage settings: " + ", ".join(missing))
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["S3_ENDPOINT"],
            region_name=os.getenv("S3_REGION", "auto"),
            aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            config=Config(s3={"addressing_style": "virtual"})
        )
        bucket = os.environ["S3_BUCKET"]
        total = len(files)
        for index, item in enumerate(files, start=1):
            path = item["path"]
            url = hf_hub_url(repo_id=REPO, filename=path, repo_type="dataset")
            with requests.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                size = item["size"] or response.headers.get("Content-Length")
                if size is None:
                    raise RuntimeError("Unknown file size for " + path)
                response.raw.decode_content = True
                s3.put_object(
                    Bucket=bucket,
                    Key="raw/horse-racing-france/" + path,
                    Body=response.raw,
                    ContentLength=int(size)
                )
            with lock:
                state["imported"].append({"path": path, "bytes": int(size)})
                state["progress"] = {"completed": index, "total": total}
        manifest = {
            "dataset": REPO, "imported_at": now(), "configs": configs,
            "files": state["imported"],
            "total_bytes": sum(x["bytes"] for x in state["imported"])
        }
        s3.put_object(
            Bucket=bucket,
            Key="raw/horse-racing-france/manifest.json",
            Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        with lock:
            state.update({"phase": "imported", "finished_at": now(), "manifest": manifest})
    except Exception as exc:
        with lock:
            state.update({"phase": "error", "error": str(exc), "finished_at": now()})

@app.get("/import")
def start_import(token: str):
    if not IMPORT_TOKEN or token != IMPORT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    with lock:
        if state["phase"] in ("importing", "imported"):
            return {"phase": state["phase"], "message": "Import already started or completed"}
        state.update({"phase": "queued", "started_at": now(), "error": None})
    threading.Thread(target=do_import, daemon=True).start()
    return {"phase": "queued", "message": "Import started; follow /status"}

@app.get("/")
def root():
    return {"message": "PMU France dataset preparation service",
            "actions": ["GET /health", "GET /inspect", "GET /status", "GET /import?token=..."]}
