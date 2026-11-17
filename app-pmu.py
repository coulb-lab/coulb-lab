import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi
REPO=os.getenv('HF_DATASET','annaelmoussa/horse-racing-france')
app=FastAPI(title='PMU France data preparation',version='0.1.0')
state={'phase':'idle','started_at':None,'finished_at':None,'files':[],'error':None}
@app.get('/health')
def health(): return {'ok':True,'service':'pmu-data-training','time':datetime.now(timezone.utc).isoformat()}
@app.get('/status')
def status(): return state
@app.post('/inspect')
def inspect():
 state.update({'phase':'inspecting','started_at':datetime.now(timezone.utc).isoformat(),'error':None})
 try:
  info=HfApi().dataset_info(REPO,files_metadata=True)
  files=[{'path':x.rfilename,'size':getattr(x,'size',None)} for x in (info.siblings or []) if x.rfilename.endswith('.parquet')]
  configs=[]
  try:
   from datasets import get_dataset_config_names
   configs=get_dataset_config_names(REPO)
  except Exception as exc: configs=['config_discovery_error: '+str(exc)[:250]]
  state.update({'phase':'inspected','finished_at':datetime.now(timezone.utc).isoformat(),'repo':REPO,'configs':configs,'files':files,'parquet_count':len(files),'total_bytes':sum(x['size'] or 0 for x in files)})
  return state
 except Exception as exc:
  state.update({'phase':'error','error':str(exc),'finished_at':datetime.now(timezone.utc).isoformat()})
  raise HTTPException(status_code=502,detail=str(exc))
@app.get('/')
def root(): return {'message':'PMU France dataset inspection service','actions':['GET /health','POST /inspect','GET /status']}
