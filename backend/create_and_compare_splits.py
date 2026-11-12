import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--dataset", default=r"..\data\travistorrent_build_level_prebuild.csv")
    p.add_argument("--output-dir", default=r"..\data")
    a = p.parse_args()
    archive, dataset, out = Path(a.archive).resolve(), Path(a.dataset).resolve(), Path(a.output_dir).resolve()
    if not archive.exists(): raise FileNotFoundError(f"Archive not found: {archive}")
    if not dataset.exists(): raise FileNotFoundError(f"Build-level dataset not found: {dataset}")

    header = pd.read_csv(archive, compression="infer", nrows=0)
    ts_col = next((c for c in ["tr_started_at", "started_at"] if c in header.columns), None)
    if ts_col is None: raise ValueError("Archive has neither tr_started_at nor started_at timestamp column.")
    if "tr_build_id" not in header.columns: raise ValueError("Archive has no tr_build_id column.")

    chunks = []
    for ch in pd.read_csv(archive, compression="infer", usecols=["tr_build_id", ts_col],
                          chunksize=200000, low_memory=False):
        ch["tr_build_id"] = pd.to_numeric(ch["tr_build_id"], errors="coerce")
        ch[ts_col] = pd.to_datetime(ch[ts_col], errors="coerce", utc=True)
        ch = ch.dropna(subset=["tr_build_id", ts_col])
        if len(ch): chunks.append(ch)
    if not chunks: raise ValueError("No valid build IDs/timestamps found.")
    raw = pd.concat(chunks, ignore_index=True)
    n_unique_times = raw.groupby("tr_build_id")[ts_col].nunique()
    ambiguous = set(n_unique_times[n_unique_times > 1].index)
    times = raw[~raw.tr_build_id.isin(ambiguous)].drop_duplicates("tr_build_id")
    times = times.rename(columns={ts_col:"build_timestamp"})[["tr_build_id","build_timestamp"]]

    df = pd.read_csv(dataset)
    if not {"tr_build_id","build_success"}.issubset(df.columns):
        raise ValueError("Build-level CSV must contain tr_build_id and build_success.")
    df["tr_build_id"] = pd.to_numeric(df["tr_build_id"], errors="coerce")
    df["build_success"] = pd.to_numeric(df["build_success"], errors="coerce")
    df = df.dropna(subset=["tr_build_id","build_success"])
    df = df[df.build_success.isin([0,1])]
    original_n = len(df)
    df = df.merge(times, on="tr_build_id", how="inner", validate="one_to_one")
    df = df.sort_values("build_timestamp").reset_index(drop=True)
    if len(df) < 100: raise ValueError(f"Only {len(df)} builds matched; insufficient for comparison.")
    features = [
    c for c in df.columns
    if c not in {
        "tr_build_id",
        "build_success",
        "build_timestamp",
        "gh_project_name"
    }
]
    if not features: raise ValueError("No model features found.")

    def evaluate(train_idx, test_idx, name):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        Xtr, Xte = tr[features], te[features]
        ytr, yte = tr.build_success.astype(int), te.build_success.astype(int)
        if ytr.nunique()!=2 or yte.nunique()!=2:
            raise ValueError(f"{name} split lacks one class in train or test; cannot calculate reliable metrics.")
        cats = [c for c in features if Xtr[c].dtype=="object"]
        nums = [c for c in features if c not in cats]
        transformers=[]
        if nums: transformers.append(("num",SimpleImputer(strategy="median"),nums))
        if cats:
            catpipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                              ("onehot",OneHotEncoder(handle_unknown="ignore"))])
            transformers.append(("cat",catpipe,cats))
        pipe=Pipeline([("prep",ColumnTransformer(transformers)),
                       ("rf",RandomForestClassifier(n_estimators=150,class_weight="balanced",random_state=42,n_jobs=-1))])
        pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)
        classes=list(pipe.named_steps["rf"].classes_)
        fail_prob=proba[:,classes.index(0)]
        actual_fail=(yte==0).astype(int); pred_fail=(pred==0).astype(int)
        return {"split":name,"train_builds":len(tr),"test_builds":len(te),
                "accuracy":float(accuracy_score(yte,pred)),
                "failure_precision":float(precision_score(actual_fail,pred_fail,zero_division=0)),
                "failure_recall":float(recall_score(actual_fail,pred_fail,zero_division=0)),
                "failure_f1":float(f1_score(actual_fail,pred_fail,zero_division=0)),
                "failure_roc_auc":float(roc_auc_score(actual_fail,fail_prob)),
                "confusion_matrix_labels_success_then_failure":confusion_matrix(yte,pred,labels=[1,0]).tolist()}

    idx=np.arange(len(df))
    rtr,rte=train_test_split(idx,test_size=.2,random_state=42,stratify=df.build_success.astype(int))
    cut=int(np.floor(len(df)*.8)); cut=min(max(cut,1),len(df)-1)
    cuttime=df.iloc[cut].build_timestamp
    ctr=np.where(df.build_timestamp < cuttime)[0]
    cte=np.where(df.build_timestamp >= cuttime)[0]
    if not len(ctr) or not len(cte): raise ValueError("Unable to form chronological split.")
    random_result=evaluate(rtr,rte,"Random stratified 80:20")
    chrono_result=evaluate(ctr,cte,"Chronological earlier-train/latest-test")
    out.mkdir(parents=True,exist_ok=True)
    results={"timestamp_column":ts_col,"archive_rows_with_valid_timestamp":len(raw),
             "unique_unambiguous_timestamped_builds":len(times),"ambiguous_build_ids_excluded":len(ambiguous),
             "build_level_rows_before_match":original_n,"matched_builds":len(df),
             "date_range_utc":[df.build_timestamp.min().isoformat(),df.build_timestamp.max().isoformat()],
             "feature_count":len(features),"features":features,
             "random_split":random_result,"chronological_split":chrono_result,
             "caution":"Results apply only to builds matched to this timestamp archive."}
    (out/"random_vs_chronological_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    pd.DataFrame([random_result,chrono_result]).to_csv(out/"random_vs_chronological_metrics.csv",index=False)
    df.to_csv(out/"timestamp_matched_build_level.csv",index=False)
    print("Timestamp column:",ts_col)
    print("Unique unambiguous timestamped builds:",len(times))
    print("Ambiguous build IDs excluded:",len(ambiguous))
    print("Build-level rows before timestamp match:",original_n)
    print("Matched builds:",len(df))
    print("Date range UTC:",results["date_range_utc"])
    for x in [random_result,chrono_result]:
        print("\\n"+x["split"])
        for k in ["train_builds","test_builds","accuracy","failure_precision","failure_recall","failure_f1","failure_roc_auc"]:
            print(f"{k}: {x[k]}")
    print("\\nSaved results under:",out)

if __name__=="__main__": main()
