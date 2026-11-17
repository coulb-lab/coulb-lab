import ast, json, os, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import ndcg_score, log_loss, brier_score_loss
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

REPO = "annaelmoussa/horse-racing-france"
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
SEED = 42

def frame(config):
    ds = load_dataset(REPO, config, split="train")
    print(f"{config}: {len(ds)} rows; columns={ds.column_names}", flush=True)
    return ds.to_pandas()

def parse_arrival(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return []
    try:
        x = ast.literal_eval(v) if isinstance(v, str) else v
        out=[]
        for item in x:
            if isinstance(item, (list, tuple)):
                out.extend(int(z) for z in item if str(z).isdigit())
            elif str(item).isdigit(): out.append(int(item))
        return out
    except Exception: return []

def main():
    t0=time.time()
    courses=frame("courses")
    runners=frame("participants")
    courses["date"]=pd.to_datetime(courses["date"], errors="coerce")
    courses=courses[courses["arriveeDefinitive"].fillna(False)].copy()
    courses["arrival_list"]=courses["ordreArrivee"].map(parse_arrival)
    courses=courses[courses["arrival_list"].map(len)>0]
    courses["year"]=courses["date"].dt.year
    courses=courses[courses["year"].between(2014,2026)]
    if "raceKey" not in runners or "numPmu" not in runners:
        raise RuntimeError("Schema participants inattendu: raceKey et numPmu sont requis.")
    runners["numPmu_num"]=pd.to_numeric(runners["numPmu"],errors="coerce")
    rank_rows=[]
    for rk, arr in zip(courses["raceKey"], courses["arrival_list"]):
        for rank, num in enumerate(arr, start=1):
            rank_rows.append((rk, int(num), rank))
    targets=pd.DataFrame(rank_rows,columns=["raceKey","numPmu_num","finish_rank"])
    data=runners.merge(targets,on=["raceKey","numPmu_num"],how="left",validate="many_to_one")
    data=data.merge(courses[["raceKey","date","year","discipline","specialite","hippodrome","distance","montantPrix","nombreDeclaresPartants","corde","categorieParticularite","conditionAge","conditionSexe"]].drop_duplicates("raceKey"),on="raceKey",how="inner",validate="many_to_one")
    # Only pre-race features. Arrival, payout and all post-race/target fields are forbidden.
    forbidden=re.compile(r"(arrivee|ordre|place|position|rang|resultat|rapport|dividende|temps_final|reduction_km|ecart_arrivee|finish|winner|target|label)",re.I)
    id_cols={"raceKey","reunionKey","numPmu","numPmu_num","date","year","finish_rank","arrival_list","ordreArrivee","arriveeDefinitive"}
    candidates=[c for c in data.columns if c not in id_cols and not forbidden.search(c)]
    # Exclude free-text identifiers likely to memorize an individual runner; retain form/history and race conditions.
    candidates=[c for c in candidates if c.lower() not in {"nom","nom_cheval","cheval","horse","raw","commentaire"}]
    # Drop mostly empty columns and unsupported nested objects.
    features=[]
    for c in candidates:
        s=data[c]
        if s.notna().mean()<0.05: continue
        if s.dtype=="object":
            sample=s.dropna().head(100)
            if len(sample) and any(isinstance(v,(dict,list,tuple)) for v in sample): continue
        features.append(c)
    if not features: raise RuntimeError("Aucune variable pré-course exploitable détectée.")
    data["relevance"]=np.where(data["finish_rank"].notna(),np.maximum(0,6-data["finish_rank"]),0).astype("int8")
    data["winner"]=(data["finish_rank"]==1).astype("int8")
    data=data.sort_values(["date","raceKey"]).reset_index(drop=True)
    # Require actual race order from the full field. Participants not in official arrival are non-top-five.
    data=data[data["year"].between(2014,2026)].copy()
    train=data[data.year<=2022].copy()
    valid=data[data.year.between(2023,2024)].copy()
    test=data[data.year>=2025].copy()
    if min(len(train),len(valid),len(test))==0: raise RuntimeError("Split chronologique vide.")
    # Keep only features with valid data; encode categorical columns consistently.
    Xtr=train[features].copy(); Xva=valid[features].copy(); Xte=test[features].copy()
    categorical=[]
    for c in features:
        if Xtr[c].dtype=="object" or str(Xtr[c].dtype)=="category" or Xtr[c].dtype=="bool":
            categorical.append(c)
            vals=pd.concat([Xtr[c],Xva[c],Xte[c]],ignore_index=True).fillna("__NA__").astype(str)
            codes,uniques=pd.factorize(vals,sort=True)
            n1=len(Xtr); n2=len(Xva)
            Xtr[c]=codes[:n1]; Xva[c]=codes[n1:n1+n2]; Xte[c]=codes[n1+n2:]
        else:
            for X in (Xtr,Xva,Xte): X[c]=pd.to_numeric(X[c],errors="coerce")
            med=Xtr[c].median()
            if pd.isna(med): med=0
            for X in (Xtr,Xva,Xte): X[c]=X[c].replace([np.inf,-np.inf],np.nan).fillna(med).astype("float32")
    # Exclude features that collapse to a single value.
    keep=[c for c in features if Xtr[c].nunique(dropna=False)>1]
    Xtr=Xtr[keep]; Xva=Xva[keep]; Xte=Xte[keep]
    categorical=[c for c in categorical if c in keep]
    ytr=train["relevance"].to_numpy(); yva=valid["relevance"].to_numpy(); yte=test["relevance"].to_numpy()
    gtr=train.groupby("raceKey",sort=False).size().to_numpy()
    gva=valid.groupby("raceKey",sort=False).size().to_numpy()
    gte=test.groupby("raceKey",sort=False).size().to_numpy()
    metrics={"dataset":REPO,"rows":{"train":len(train),"validation":len(valid),"test":len(test)},
             "races":{"train":train.raceKey.nunique(),"validation":valid.raceKey.nunique(),"test":test.raceKey.nunique()},
             "features":keep,"feature_count":len(keep),"date_split":"train<=2022; validation=2023-2024; test>=2025",
             "target":"official ordreArrivee parsed to finish_rank; relevance=max(0,6-rank)",
             "leakage_exclusions":"arrival/order/place/rank/result/payout/post-race fields excluded"}
    # Baseline: if a pre-race odds field exists, lower odds are better; otherwise mark unavailable.
    odds=[c for c in keep if re.search(r"(cote|odds|rapport_probable)",c,re.I)]
    metrics["odds_baseline"]="not_available" if not odds else {"candidate_columns":odds}
    # LightGBM LambdaMART, primary ranking model.
    try:
        import lightgbm as lgb
        ranker=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",n_estimators=500,
            learning_rate=0.04,num_leaves=31,max_depth=-1,min_child_samples=80,
            subsample=0.85,colsample_bytree=0.85,reg_lambda=2.0,random_state=SEED,
            n_jobs=-1,verbosity=-1)
        ranker.fit(Xtr,ytr,group=gtr,eval_set=[(Xva,yva)],eval_group=[gva],
                   eval_at=[1,3,5],callbacks=[lgb.early_stopping(40,verbose=False)])
        pred=ranker.predict(Xte)
        metrics["LightGBM_LambdaMART"]={"best_iteration":ranker.best_iteration_}
        model_preds={"LightGBM_LambdaMART":pred}
    except Exception as e:
        metrics["LightGBM_LambdaMART"]={"error":repr(e)}
        model_preds={}
    # Binary top-five classifier baseline model; bounded sample for memory and runtime.
    y5=(test["finish_rank"].notna()).astype("int8").to_numpy()
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        rng=np.random.default_rng(SEED)
        idx=np.arange(len(train))
        if len(idx)>450000: idx=rng.choice(idx,450000,replace=False)
        clf=ExtraTreesClassifier(n_estimators=250,min_samples_leaf=4,max_features="sqrt",
                                 class_weight="balanced_subsample",n_jobs=-1,random_state=SEED)
        clf.fit(Xtr.iloc[idx],(train.iloc[idx]["finish_rank"].notna()).astype("int8"))
        p=clf.predict_proba(Xte)[:,1]
        model_preds["ExtraTrees_top5_probability"]=p
        metrics["ExtraTrees_top5_probability"]={"train_sample_rows":len(idx)}
    except Exception as e: metrics["ExtraTrees_top5_probability"]={"error":repr(e)}
    # Evaluate ranking per race.
    def eval_pred(pred):
        tmp=test[["raceKey","finish_rank"]].copy(); tmp["score"]=pred
        nd=[]; top1=0; top5=0; hit_top5=0; n=0
        for _,g in tmp.groupby("raceKey",sort=False):
            if len(g)<2: continue
            rel=np.where(g.finish_rank.notna(),np.maximum(0,6-g.finish_rank.fillna(99)),0)
            s=g.score.to_numpy()
            order=np.argsort(-s)
            top1+=int(g.iloc[order[0]].finish_rank==1)
            top5+=int(np.any(g.iloc[order[:5]].finish_rank==1))
            hit_top5+=int(np.sum(g.iloc[order[:5]].finish_rank.notna()))
            nd.append(ndcg_score([rel],[s],k=5))
            n+=1
        return {"races":n,"winner_hit_rate":top1/n if n else None,
                "winner_in_predicted_top5_rate":top5/n if n else None,
                "top5_precision":hit_top5/(5*n) if n else None,
                "NDCG_at_5":float(np.mean(nd)) if nd else None}
    for name,pred in model_preds.items():
        if len(pred)==len(test): metrics[name].update(eval_pred(pred))
    # Persist reproducibility details and results.
    metrics["elapsed_seconds"]=round(time.time()-t0,1)
    with open(OUT/"metrics.json","w") as f: json.dump(metrics,f,indent=2,ensure_ascii=False,default=str)
    print(json.dumps(metrics,indent=2,ensure_ascii=False,default=str),flush=True)

if __name__=="__main__": main()
