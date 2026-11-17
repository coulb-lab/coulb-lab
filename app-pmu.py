import json
import io
import logging
import os
import threading
from datetime import datetime, timezone

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import HfApi, hf_hub_url
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pmu-import")

REPO = os.getenv("HF_DATASET", "annaelmoussa/horse-racing-france")
IMPORT_TOKEN = os.getenv("IMPORT_TOKEN", "")
PREFIX = "raw/horse-racing-france/"
STATE_KEY = PREFIX + "import-state.json"
MANIFEST_KEY = PREFIX + "manifest.json"
app = FastAPI(title="PMU France data preparation", version="0.3.0")
lock = threading.Lock()
state = {
    "phase": "idle", "started_at": None, "finished_at": None,
    "repo": REPO, "files": [], "imported": [], "error": None
}

def now():
    return datetime.now(timezone.utc).isoformat()

def s3_client():
    required = ["S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError("Missing storage settings: " + ", ".join(missing))
    return boto3.client(
        "s3", endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(s3={"addressing_style": "virtual"})
    )

def persist_state(s3=None):
    s3 = s3 or s3_client()
    s3.put_object(
        Bucket=os.environ["S3_BUCKET"], Key=STATE_KEY,
        Body=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

def read_json_object(s3, key):
    try:
        obj = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise

def load_persisted_state():
    try:
        persisted = read_json_object(s3_client(), STATE_KEY)
        if persisted:
            state.update(persisted)
    except Exception as exc:
        log.warning("Persistent state unavailable: %s", str(exc)[:300])
    return state

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
    load_persisted_state()
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
            persist_state()
        return state
    except Exception as exc:
        with lock:
            state.update({"phase": "error", "error": str(exc)[:1000], "finished_at": now()})
            try: persist_state()
            except Exception: pass
        log.exception("Dataset inspection failed")
        raise HTTPException(status_code=502, detail="Dataset inspection failed; see service logs")

def do_audit_background():
    try:
        s3 = s3_client()
        manifest = read_json_object(s3, MANIFEST_KEY)
        if not manifest:
            log.warning("Automatic audit skipped: dataset manifest missing")
            return
        tables, errors, total_rows = {}, [], 0
        for item in manifest.get("files", []):
            path = item["path"]
            body = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=PREFIX + path)["Body"].read()
            parquet = pq.ParquetFile(io.BytesIO(body))
            table_name = path.split("/")[0] if "/" in path else "default"
            entry = tables.setdefault(table_name, {"files": 0, "rows": 0, "bytes": 0,
                                                    "columns": None, "schema": None, "row_groups": 0})
            schema = parquet.schema_arrow
            columns = schema.names
            if entry["columns"] is None:
                entry["columns"] = columns
                entry["schema"] = {field.name: str(field.type) for field in schema}
            elif entry["columns"] != columns:
                errors.append({"file": path, "error": "inconsistent_columns"})
            entry["files"] += 1
            entry["rows"] += parquet.metadata.num_rows
            entry["bytes"] += len(body)
            entry["row_groups"] += parquet.metadata.num_row_groups
            total_rows += parquet.metadata.num_rows
            if table_name in ("courses", "participants", "rapports") and "raceKey" not in columns:
                errors.append({"file": path, "error": "missing_raceKey"})
        report = {
            "audited_at": now(), "dataset": REPO,
            "manifest_files": len(manifest.get("files", [])),
            "manifest_bytes": manifest.get("total_bytes"),
            "audited_bytes": sum(x["bytes"] for x in tables.values()),
            "total_rows": total_rows, "tables": tables, "errors": errors,
            "status": "schema_ok" if not errors else "schema_errors",
            "limitations": ["Schema/readability/row-count audit only; exact key uniqueness and referential integrity require a second row-level pass."]
        }
        s3.put_object(Bucket=os.environ["S3_BUCKET"], Key=PREFIX + "audit-report.json",
                      Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                      ContentType="application/json")
        log.info("Automatic dataset audit completed: %s files, %s rows, %s tables, %s errors",
                 report["manifest_files"], total_rows, len(tables), len(errors))
    except Exception:
        log.exception("Automatic dataset audit failed")

def do_import():
    s3 = None
    try:
        s3 = s3_client()
        previous = read_json_object(s3, STATE_KEY) or {}
        already = {x["path"]: x for x in previous.get("imported", []) if x.get("path")}
        with lock:
            state.update(previous)
            state.update({"phase": "importing", "started_at": state.get("started_at") or now(),
                          "finished_at": None, "repo": REPO, "error": None})
            state["imported"] = list(already.values())
            persist_state(s3)
        files, configs = inspect_dataset()
        total = len(files)
        for index, item in enumerate(files, start=1):
            path = item["path"]
            key = PREFIX + path
            try:
                head = s3.head_object(Bucket=os.environ["S3_BUCKET"], Key=key)
                expected = item.get("size")
                if expected is not None and int(head.get("ContentLength", -1)) != int(expected):
                    raise RuntimeError("Stored object size mismatch for " + path)
                stored_bytes = int(head["ContentLength"])
                already[path] = {"path": path, "bytes": stored_bytes, "verified": True}
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in ("404", "NoSuchKey", "NotFound"):
                    raise
                url = hf_hub_url(repo_id=REPO, filename=path, repo_type="dataset")
                with requests.get(url, stream=True, timeout=(30, 900)) as response:
                    response.raise_for_status()
                    size = item.get("size") or response.headers.get("Content-Length")
                    if size is None:
                        raise RuntimeError("Unknown file size for " + path)
                    response.raw.decode_content = True
                    s3.put_object(Bucket=os.environ["S3_BUCKET"], Key=key,
                                  Body=response.raw, ContentLength=int(size))
                head = s3.head_object(Bucket=os.environ["S3_BUCKET"], Key=key)
                stored_bytes = int(head["ContentLength"])
                if item.get("size") is not None and stored_bytes != int(item["size"]):
                    raise RuntimeError("Post-upload size mismatch for " + path)
                already[path] = {"path": path, "bytes": stored_bytes, "verified": True}
            with lock:
                state.update({"phase": "importing", "configs": configs, "files": files,
                              "imported": list(already.values()),
                              "progress": {"completed": len(already), "total": total,
                                           "current": path}, "error": None})
                persist_state(s3)
            log.info("Parquet file verified %s (%s/%s)", path, index, total)

        manifest = {
            "dataset": REPO, "imported_at": now(), "configs": configs,
            "files": sorted(already.values(), key=lambda x: x["path"]),
            "total_files": len(already),
            "total_bytes": sum(x["bytes"] for x in already.values())
        }
        s3.put_object(Bucket=os.environ["S3_BUCKET"], Key=MANIFEST_KEY,
                      Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                      ContentType="application/json")
        with lock:
            state.update({"phase": "imported", "finished_at": now(),
                          "manifest": manifest, "progress": {"completed": total, "total": total},
                          "error": None})
            persist_state(s3)
        log.info("Dataset import completed: %s files, %s bytes", total, manifest["total_bytes"])
    except Exception as exc:
        with lock:
            state.update({"phase": "error", "error": str(exc)[:1000], "finished_at": now()})
            try:
                if s3: persist_state(s3)
            except Exception as persist_exc:
                log.error("Could not persist failure state: %s", str(persist_exc)[:300])
        log.exception("Dataset import failed")

def start_import_authorized():
    load_persisted_state()
    with lock:
        if state.get("phase") == "imported":
            return {"phase": "imported", "message": "Dataset already imported"}
        if state.get("phase") == "importing":
            return {"phase": "importing", "message": "Import already running"}
        state.update({"phase": "queued", "started_at": now(), "finished_at": None, "error": None})
        persist_state()
    threading.Thread(target=do_import, daemon=True).start()
    return {"phase": "queued", "message": "Import started; follow /status"}

@app.post("/import/start")
def start_import_header(x_import_token: str = Header(default="")):
    if not IMPORT_TOKEN or x_import_token != IMPORT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return start_import_authorized()

@app.get("/import")
def start_import(token: str):
    if not IMPORT_TOKEN or token != IMPORT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return start_import_authorized()

@app.get("/verify")
def verify_storage():
    try:
        s3 = s3_client()
        manifest = read_json_object(s3, MANIFEST_KEY)
        response = s3.list_objects_v2(Bucket=os.environ["S3_BUCKET"], Prefix=PREFIX)
        objects = [{"key": x["Key"], "bytes": x["Size"]} for x in response.get("Contents", [])]
        return {"manifest_present": manifest is not None, "manifest": manifest,
                "objects_seen_in_first_page": len(objects), "objects": objects}
    except Exception as exc:
        log.exception("Storage verification failed")
        raise HTTPException(status_code=502, detail="Storage verification failed; see service logs")


@app.post("/audit")
def audit_dataset(x_import_token: str = Header(default="")):
    if not IMPORT_TOKEN or x_import_token != IMPORT_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        s3 = s3_client()
        manifest = read_json_object(s3, MANIFEST_KEY)
        if not manifest:
            raise HTTPException(status_code=409, detail="Dataset manifest is not available")
        tables = {}
        errors = []
        total_rows = 0
        for item in manifest.get("files", []):
            path = item["path"]
            key = PREFIX + path
            body = s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=key)["Body"].read()
            parquet = pq.ParquetFile(io.BytesIO(body))
            table_name = path.split("/")[0] if "/" in path else "default"
            entry = tables.setdefault(table_name, {
                "files": 0, "rows": 0, "bytes": 0, "columns": None,
                "schema": None, "row_groups": 0
            })
            schema = parquet.schema_arrow
            columns = schema.names
            if entry["columns"] is None:
                entry["columns"] = columns
                entry["schema"] = {field.name: str(field.type) for field in schema}
            elif entry["columns"] != columns:
                errors.append({"file": path, "error": "inconsistent_columns"})
            entry["files"] += 1
            entry["rows"] += parquet.metadata.num_rows
            entry["bytes"] += len(body)
            entry["row_groups"] += parquet.metadata.num_row_groups
            total_rows += parquet.metadata.num_rows
            required_key = "raceKey" if table_name in ("courses", "participants", "rapports") else None
            if required_key and required_key not in columns:
                errors.append({"file": path, "error": "missing_raceKey"})
        report = {
            "audited_at": now(), "dataset": REPO,
            "manifest_files": len(manifest.get("files", [])),
            "manifest_bytes": manifest.get("total_bytes"),
            "audited_bytes": sum(x["bytes"] for x in tables.values()),
            "total_rows": total_rows, "tables": tables,
            "errors": errors,
            "status": "schema_ok" if not errors else "schema_errors",
            "limitations": [
                "This first pass validates Parquet readability, schema consistency, row counts and key-column presence.",
                "It does not yet prove uniqueness of raceKey or referential integrity across all rows."
            ]
        }
        s3.put_object(Bucket=os.environ["S3_BUCKET"], Key=PREFIX + "audit-report.json",
                      Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                      ContentType="application/json")
        log.info("Dataset audit completed: %s files, %s rows, %s tables, %s schema errors",
                 report["manifest_files"], total_rows, len(tables), len(errors))
        return report
    except HTTPException:
        raise
    except Exception:
        log.exception("Dataset audit failed")
        raise HTTPException(status_code=502, detail="Dataset audit failed; see service logs")

@app.get("/")
def root():
    return {"message": "PMU France dataset preparation service",
            "actions": ["GET /health", "GET /inspect", "GET /status",
                        "POST /import/start (X-Import-Token header)", "GET /verify"]}


@app.on_event("startup")
def resume_dataset_import():
    """Automatically resume the one-time dataset import after deployment/restart."""
    try:
        s3 = s3_client()
        manifest = read_json_object(s3, MANIFEST_KEY)
        if manifest:
            files = manifest.get("files", [])
            missing = []
            verified_bytes = 0
            for item in files:
                key = PREFIX + item["path"]
                try:
                    head = s3.head_object(Bucket=os.environ["S3_BUCKET"], Key=key)
                    actual = int(head.get("ContentLength", -1))
                    expected = int(item.get("bytes", -1))
                    if actual != expected:
                        missing.append({"path": item["path"], "reason": "size_mismatch",
                                        "expected": expected, "actual": actual})
                    else:
                        verified_bytes += actual
                except Exception as exc:
                    missing.append({"path": item.get("path"), "reason": str(exc)[:250]})
            state.update({
                "phase": "imported" if not missing else "verification_error",
                "manifest": manifest, "imported": files,
                "progress": {"completed": len(files) - len(missing),
                             "total": len(files)},
                "finished_at": manifest.get("imported_at"),
                "verification": {"verified_files": len(files) - len(missing),
                                 "missing_or_invalid": missing,
                                 "verified_bytes": verified_bytes}
            })
            log.info("Persistent dataset audit: %s/%s files verified; %s bytes; %s invalid",
                     len(files) - len(missing), len(files), verified_bytes, len(missing))
            if missing:
                log.error("Dataset manifest verification failures: %s", json.dumps(missing)[:2000])
            else:
                log.info("All manifest objects verified; import will not be repeated")
            try:
                audit_obj = s3.head_object(Bucket=os.environ["S3_BUCKET"], Key=PREFIX + "audit-report.json")
                log.info("Existing dataset audit report found (%s bytes)", audit_obj.get("ContentLength"))
            except Exception:
                threading.Thread(target=do_audit_background, daemon=True).start()
                log.info("Automatic dataset schema audit started")
            return
        previous = read_json_object(s3, STATE_KEY) or {}
        state.update(previous)
        if state.get("phase") == "importing":
            log.info("Resuming interrupted dataset import from persistent progress")
        else:
            state.update({"phase": "queued", "started_at": now(),
                          "finished_at": None, "error": None})
        persist_state(s3)
        threading.Thread(target=do_import, daemon=True).start()
        log.info("Dataset import/resume task started automatically")
    except Exception:
        log.exception("Automatic dataset import could not start")
