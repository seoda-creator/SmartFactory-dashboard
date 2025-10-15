# -*- coding: utf-8 -*-

# ================= Determinism Guard (put BEFORE other imports) =================
import os as _os, random as _random, numpy as _np
_os.environ["PYTHONHASHSEED"] = "0"
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"
_random.seed(42)
_np.random.seed(42)
# ==============================================================================

import os, re, numpy as np, pandas as pd
from collections import Counter

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_curve
from imblearn.over_sampling import RandomOverSampler, ADASYN
from xgboost import XGBClassifier

# ================= 경로/옵션 =================
TRAIN_DIR = "./train_ml_imputed"
TEST_DIR  = "./test_ml_imputed"
OUT_DIR   = "./yq_cls_out"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
CANDIDATE_N_SPLITS = [3,4,5,6,7,8]
ALPHA = 1.0

# 분류 시드 앙상블
CLS_ENSEMBLE_SEEDS = [0,1,2,3,4]

# 불량 가중(라인/제품별)
DEFECT_BOOST_DEFAULT_A31 = 1.3
DEFECT_BOOST_DEFAULT_N31 = 2.8
DEFECT_BOOST_MAP = {
    ("T010305","A_31"): 1.3,
    ("T010306","A_31"): 0.9,
    ("T050304","A_31"): 2.25,
    ("T050307","A_31"): 1.0,
    ("T100304","N_31"): 1.8,
    ("T100306","N_31"): 2.4
}

# 임계값 전략
STRATEGY = 1
USE_CAP_FOR_N31 = True
CAP_VALUE = 0.55

# ================= 유틸 =================
def combo_from_name(fn: str):
    m = re.search(r"(T\d+).*(A_31|N_31)", os.path.splitext(fn)[0])
    return (m.group(1), m.group(2)) if m else ("UNK","UNK")

def get_feat_cols(df):
    return [c for c in df.columns if isinstance(c, str) and c.startswith("C_")]

def get_defect_boost(line: str, product: str) -> float:
    if (line, product) in DEFECT_BOOST_MAP:
        return DEFECT_BOOST_MAP[(line, product)]
    return DEFECT_BOOST_DEFAULT_N31 if product == "N_31" else DEFECT_BOOST_DEFAULT_A31

def pretty_print_table(df, title=None):
    if title: print(f"\n=== {title} ===")
    with pd.option_context('display.max_columns', None,
                           'display.width', 140,
                           'display.max_colwidth', 60,
                           'display.precision', 6):
        print(df.to_string(index=False))

# ================= 불균형/샘플링 =================
def make_sample_weights(y, alpha, defect_boost):
    cls, cnt = np.unique(y, return_counts=True)
    med = np.median(cnt)
    w_map = {}
    for c, n in zip(cls, cnt):
        base = (med / max(1, n)) ** alpha
        if int(c) in (0, 2): base *= defect_boost
        w_map[int(c)] = float(base)
    sw = np.array([w_map[int(t)] for t in y], dtype=float)
    return sw, w_map

def resample_for_classification(X_tr, y_tr, random_state=RANDOM_STATE):
    cls, cnt = np.unique(y_tr, return_counts=True)
    if len(cnt) < 2: return X_tr, y_tr, "skip", len(X_tr), 0
    maj = cnt.max(); minc = cnt.min(); imb = maj / max(1, minc)
    method = "skip"; X_rs, y_rs = X_tr, y_tr
    try:
        if imb <= 1.8:
            method = "skip"
        elif imb <= 3.0:
            ros = RandomOverSampler(random_state=random_state)
            X_rs, y_rs = ros.fit_resample(X_tr, y_tr); method = "ros"
        else:
            if minc >= 5:
                ada = ADASYN(random_state=random_state, n_neighbors=3)
                X_rs, y_rs = ada.fit_resample(X_tr, y_tr); method = "adasyn"
            else:
                ros = RandomOverSampler(random_state=random_state)
                X_rs, y_rs = ros.fit_resample(X_tr, y_tr); method = "ros"
    except ValueError:
        try:
            ros = RandomOverSampler(random_state=random_state)
            X_rs, y_rs = ros.fit_resample(X_tr, y_tr); method = "ros"
        except ValueError:
            method, X_rs, y_rs = "skip", X_tr, y_tr
    return X_rs, y_rs, method, len(X_tr), (len(X_rs)-len(X_tr))

# ================= 폴드 자동탐색 =================
def select_best_n_splits(X_raw, y_class, defect_boost, candidates, random_state=RANDOM_STATE):
    results = []
    proxy_xgb = XGBClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        subsample=0.9, colsample_bytree=0.9,
        random_state=random_state, tree_method="hist", n_jobs=-1,
        eval_metric="mlogloss", objective="multi:softprob", num_class=3
    )
    for n in candidates:
        cls, cnt = np.unique(y_class, return_counts=True)
        if (cnt.min() < n): continue
        skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state)
        oof_t, oof_p = [], []
        for tr_idx, va_idx in skf.split(X_raw, y_class):
            Xtr, Xva = X_raw[tr_idx], X_raw[va_idx]
            ytr, yva = y_class[tr_idx], y_class[va_idx]
            Xfit, yfit, *_ = resample_for_classification(Xtr, ytr, random_state=random_state)
            sw, _ = make_sample_weights(yfit, ALPHA, defect_boost)
            proxy_xgb.fit(Xfit, yfit, sample_weight=sw)
            pred = proxy_xgb.predict(Xva)
            oof_t.extend(yva.tolist()); oof_p.extend(pred.tolist())
        macro = classification_report(oof_t, oof_p, digits=4, output_dict=True)["macro avg"]["f1-score"]
        results.append((n, macro))
    if not results: return DEFAULT_N_SPLITS, []
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[0][0], results

# ================= 임계값/전략 =================
def roc_best_threshold(y_true_binary, prob_pos):
    fpr, tpr, thr = roc_curve(y_true_binary, prob_pos)
    if thr is None or len(thr)==0: return 0.5
    j = tpr - fpr
    return float(thr[np.argmax(j)])

def f1_best_threshold(y_true_binary, prob_pos, grid=None):
    y = np.asarray(y_true_binary).astype(int)
    p = np.asarray(prob_pos).astype(float)
    if grid is None: grid = np.unique(np.concatenate([np.linspace(0.0,1.0,101), p]))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def prevalence_threshold(prob_pos, desired_rate):
    desired = float(np.clip(desired_rate, 0.01, 0.99))
    return float(np.quantile(prob_pos, 1.0 - desired))

def _thr_by_method(method, y_true_binary, score_vec):
    method = method.lower()
    if method == "roc": return roc_best_threshold(y_true_binary, score_vec)
    elif method == "f1": return f1_best_threshold(y_true_binary, score_vec)[0]
    elif method == "preval":
        prev = float(np.mean(y_true_binary))
        desired = max(0.05, min(0.5, prev*0.9))
        return prevalence_threshold(score_vec, desired_rate=desired)
    else: raise ValueError("unknown method:", method)

def decide_thresholds_by_strategy(strategy, product, y_val, prob_val,
                                  use_cap_for_n31=True, cap_value=0.55):
    """불량(0/2) vs 정상(1) → thr1, 불량 내부 0 vs 2 → thr2"""
    p0 = prob_val[:,0]; p2 = prob_val[:,2]
    p_def = p0 + p2
    y_bin_def = pd.Series(y_val).map({1:0, 0:1, 2:1}).astype(int).values
    m02 = np.isin(y_val, [0,2]); has_02 = (m02.sum() > 0 and np.unique(y_val[m02]).size > 1)
    if has_02:
        y02 = pd.Series(y_val[m02]).map({0:0, 2:1}).astype(int).values
        score02 = p2[m02]
    else:
        y02 = None; score02 = None

    if strategy == 1:      thr1 = _thr_by_method("roc",   y_bin_def, p_def)
    elif strategy == 2:    thr1 = _thr_by_method("f1",    y_bin_def, p_def)
    elif strategy in (3,4):
        cand = [_thr_by_method("roc", y_bin_def, p_def),
                _thr_by_method("f1",  y_bin_def, p_def),
                _thr_by_method("preval", y_bin_def, p_def)]
        def _f1(yb,s,t): return f1_score(yb, (s>=t).astype(int), zero_division=0)
        thr1 = cand[int(np.argmax([_f1(y_bin_def,p_def,t) for t in cand]))]
    else: raise ValueError("Unknown STRATEGY")

    if use_cap_for_n31 and product == "N_31":
        thr1 = min(thr1, cap_value)

    if not has_02:
        thr2 = 0.5
    else:
        if strategy == 1:      thr2 = _thr_by_method("roc", y02, score02)
        elif strategy == 2:    thr2 = _thr_by_method("f1",  y02, score02)
        elif strategy == 3:    thr2 = _thr_by_method("roc", y02, score02)
        elif strategy == 4:
            cand = [_thr_by_method("roc", y02, score02),
                    _thr_by_method("f1",  y02, score02),
                    _thr_by_method("preval", y02, score02)]
            def _f1(yb,s,t): return f1_score(yb, (s>=t).astype(int), zero_division=0)
            thr2 = cand[int(np.argmax([_f1(y02,score02,t) for t in cand]))]
        else: raise ValueError("Unknown STRATEGY")
    return float(thr1), float(thr2)

# ================= 분류 OOF/예측 =================
def oof_classification_direct(X, yc, cls_params, defect_boost, product, strategy_local, n_splits_global):
    skf = StratifiedKFold(n_splits=n_splits_global, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(yc, int)
    for tr, va in skf.split(X, yc):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = yc[tr], yc[va]
        Xfit, yfit, *_ = resample_for_classification(Xtr, ytr, random_state=RANDOM_STATE)
        sw, _ = make_sample_weights(yfit, ALPHA, defect_boost)
        clf = XGBClassifier(**cls_params).fit(Xfit, yfit, sample_weight=sw)
        prob = clf.predict_proba(Xva)
        thr1, thr2 = decide_thresholds_by_strategy(strategy_local, product, yva, prob,
                                                   use_cap_for_n31=USE_CAP_FOR_N31, cap_value=CAP_VALUE)
        p_def = prob[:,0] + prob[:,2]
        is_def = (p_def >= thr1)
        pred02 = np.where(prob[:,2] >= thr2, 2, 0)
        y_pred[va] = np.where(is_def, pred02, 1).astype(int)
    return y_pred

# 분류 OOF 확률 — 시드 앙상블 평균
def oof_classification_proba(X, yc, cls_params, defect_boost, seeds, n_splits_global):
    skf = StratifiedKFold(n_splits=n_splits_global, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_accum = np.zeros((len(yc), 3), dtype=float)
    for seed in seeds:
        params = cls_params.copy()
        params["random_state"] = seed
        oof_prob = np.zeros((len(yc), 3), dtype=float)
        for tr, va in skf.split(X, yc):
            Xtr, Xva = X[tr], X[va]; ytr = yc[tr]
            Xfit, yfit, *_ = resample_for_classification(Xtr, ytr, random_state=seed)
            sw, _ = make_sample_weights(yfit, ALPHA, defect_boost)
            clf = XGBClassifier(**params).fit(Xfit, yfit, sample_weight=sw)
            oof_prob[va] = clf.predict_proba(Xva)
        oof_prob_accum += oof_prob
    return oof_prob_accum / max(1, len(seeds))

# ================= 메인(단일 DF) =================
def run_one_df(train_csv, test_csv=None, cls_params=None, strategy=None):
    base = os.path.basename(train_csv)
    line, product = combo_from_name(base)

    df = pd.read_csv(train_csv)
    assert {"Y_Quality","Y_Class"}.issubset(df.columns), "라벨 누락"
    feats = get_feat_cols(df); assert len(feats)>0, "C_* 피처 없음"

    X  = df[feats].astype(float).values
    yc = df["Y_Class"].astype(int).values

    defect_boost = get_defect_boost(line, product)

    # 폴드 자동 선택
    best_splits, tried = select_best_n_splits(X, yc, defect_boost, CANDIDATE_N_SPLITS, random_state=RANDOM_STATE)
    n_splits_global = best_splits

    STRATEGY_LOCAL = strategy if strategy is not None else STRATEGY

    CLS = dict(
        objective="multi:softprob", eval_metric="mlogloss", num_class=3, tree_method="hist",
        n_estimators=500, learning_rate=0.08, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1
    )
    if cls_params: CLS.update(cls_params)

    print(f"\n================ {base} | Train={len(df)} | Feats={len(feats)} "
          f"| Best n_splits={best_splits} | strategy={STRATEGY_LOCAL} ================")
    if tried:
        tried_df = pd.DataFrame(tried, columns=["n_splits","macroF1"]).sort_values("n_splits").reset_index(drop=True)
        print("Tried splits (n, macroF1):", list(map(tuple, tried_df.values)))

    # 1) 분류 OOF 확률(시드 앙상블)
    prob_oof = oof_classification_proba(X, yc, CLS, defect_boost, seeds=CLS_ENSEMBLE_SEEDS,
                                        n_splits_global=n_splits_global)

    # 2) 직접 분류 OOF (임계 전략 적용)
    y_cls_oof = oof_classification_direct(X, yc, CLS, defect_boost, product,
                                          strategy_local=STRATEGY_LOCAL, n_splits_global=n_splits_global)
    print("[Direct-Clf OOF]")
    print(classification_report(yc, y_cls_oof, digits=4))

    # OOF 저장 (라벨 포함)
    out_base = os.path.splitext(base)[0]
    df = df.reset_index(drop=True)
    oof_save = df[["Y_Class"]].copy()    
    oof_save["y_direct_clf"] = y_cls_oof
    oof_save["meta_best_n_splits"] = n_splits_global
    oof_save["meta_strategy"] = STRATEGY_LOCAL
    oof_fp = os.path.join(OUT_DIR, f"oof_summary_{out_base}.csv")
    oof_save.to_csv(oof_fp, index=False, encoding="utf-8-sig")
    print(f"[저장] OOF Summary → {oof_fp}")

    # ================= 테스트 예측 =================
    if test_csv is not None and os.path.exists(test_csv):
        dft = pd.read_csv(test_csv).reindex(columns=feats, fill_value=0.0)
        Xt  = dft[feats].astype(float).values

        # 분류 full (시드 앙상블 확률 평균)
        prob_test_accum = np.zeros((len(Xt), 3), dtype=float)
        prob_train_accum = np.zeros((len(X), 3), dtype=float)
        for seed in CLS_ENSEMBLE_SEEDS:
            params = CLS.copy(); params["random_state"] = seed
            X_fit_c, y_fit_c, *_ = resample_for_classification(X, yc, random_state=seed)
            sw_full, _ = make_sample_weights(y_fit_c, ALPHA, defect_boost)
            clf = XGBClassifier(**params).fit(X_fit_c, y_fit_c, sample_weight=sw_full)
            prob_train_accum += clf.predict_proba(X)
            prob_test_accum  += clf.predict_proba(Xt)
        prob_train = prob_train_accum / max(1, len(CLS_ENSEMBLE_SEEDS))
        prob_test  = prob_test_accum  / max(1, len(CLS_ENSEMBLE_SEEDS))

        # (직접분류) 전략 기반 임계 (train 기준으로 결정 → test 적용)
        thr1_all, thr2_all = decide_thresholds_by_strategy(
            STRATEGY_LOCAL, product, yc, prob_train,
            use_cap_for_n31=USE_CAP_FOR_N31, cap_value=CAP_VALUE
        )
        p_def_te  = prob_test[:,0] + prob_test[:,2]
        is_def_te = (p_def_te >= thr1_all)
        pred02_te = np.where(prob_test[:,2] >= thr2_all, 2, 0)
        y_pred_direct = np.where(is_def_te, pred02_te, 1).astype(int)

        # 저장
        df_out = dft.copy()
        df_out["y_direct_clf"] = y_pred_direct
        df_out["prob_cls_0"]   = prob_test[:,0]
        df_out["prob_cls_1"]   = prob_test[:,1]
        df_out["prob_cls_2"]   = prob_test[:,2]
        df_out["thr1_def_vs_norm"] = thr1_all
        df_out["thr2_0_vs_2"]      = thr2_all

        save_fp = os.path.join(OUT_DIR, f"test_pred_{out_base}.csv")
        df_out.to_csv(save_fp, index=False, encoding="utf-8-sig")
        print(f"[저장] TEST 예측(분류 단독) → {save_fp}")
    else:
        print("[TEST 없음: 예측 저장 생략]")

# DF1
cls_params = dict(
    objective="multi:softprob", eval_metric="mlogloss", num_class=3,
    n_estimators=190, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=1.0, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1
)
train_fp = os.path.join(TRAIN_DIR, "T010305_A_31_ml_ready.csv")
test_fp  = os.path.join(TEST_DIR,  "test_ml_T010305_A_31.csv")
run_one_df(train_fp, test_fp, cls_params=cls_params, strategy=1)

# DF2
cls_params = dict(
    objective="multi:softprob", eval_metric="mlogloss", num_class=3,
    n_estimators=180, learning_rate=0.12, max_depth=6,
    subsample=0.9, colsample_bytree=0.85, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
    min_child_weight=5, gamma=0.1, reg_alpha=0.5, reg_lambda=1.0
)
train_fp = os.path.join(TRAIN_DIR, "T010306_A_31_ml_ready.csv")
test_fp  = os.path.join(TEST_DIR,  "test_ml_T010306_A_31.csv")
run_one_df(train_fp, test_fp, cls_params=cls_params, strategy=1)

# DF3
cls_params = dict(
    objective="multi:softprob", eval_metric="mlogloss", num_class=3,
    n_estimators=200, learning_rate=0.07, max_depth=6,
    subsample=0.8, colsample_bytree=0.9, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
    min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.5
)
train_fp = os.path.join(TRAIN_DIR, "T050304_A_31_ml_ready.csv")
test_fp  = os.path.join(TEST_DIR,  "test_ml_T050304_A_31.csv")
run_one_df(train_fp, test_fp, cls_params=cls_params, strategy=1)

# DF4
cls_params = dict(
    objective="multi:softprob", eval_metric="mlogloss", num_class=3,
    n_estimators=50, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=1.0, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
    min_child_weight=1, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0
)
train_fp = os.path.join(TRAIN_DIR, "T050307_A_31_ml_ready.csv")
test_fp  = os.path.join(TEST_DIR,  "test_ml_T050307_A_31.csv")
run_one_df(train_fp, test_fp, cls_params=cls_params, strategy=1)

# DF5
cls_params = dict(
    objective="multi:softprob", eval_metric="mlogloss", num_class=3,
    n_estimators=200, learning_rate=0.09, max_depth=3,
    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
    min_child_weight=1, gamma=0.0, reg_alpha=0.1, reg_lambda=2.0
)
train_fp = os.path.join(TRAIN_DIR, "T100304_N_31_ml_ready.csv")
test_fp  = os.path.join(TEST_DIR,  "test_ml_T100304_N_31.csv")
run_one_df(train_fp, test_fp, cls_params=cls_params, strategy=1)

# DF6
cls_params = dict(
    objective="multi:softprob", eval_metric="mlogloss", num_class=3,
    n_estimators=130, learning_rate=0.09, max_depth=3,
    subsample=0.8, colsample_bytree=0.7, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
    min_child_weight=2, gamma=0.0, reg_alpha=0.1, reg_lambda=1.5
)
train_fp = os.path.join(TRAIN_DIR, "T100306_N_31_ml_ready.csv")
test_fp  = os.path.join(TEST_DIR,  "test_ml_T100306_N_31.csv")
run_one_df(train_fp, test_fp, cls_params=cls_params, strategy=1)