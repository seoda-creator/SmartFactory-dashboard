# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import time
from pathlib import Path
from datetime import datetime
# 필요 시만 사용: pages에서 쓰면 requirements.txt에 streamlit-plotly-events 추가
# from streamlit_plotly_events import plotly_events
import re

# ==============================
# GitHub/Cloud용 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent

# 데이터 폴더(리포지터리 내부)
TRAIN_DIR = BASE_DIR / "train_ml_imputed"
TEST_DIR  = BASE_DIR / "test_ml_imputed"
DATA_DIR  = TRAIN_DIR  # 기존 코드 호환

# 매핑 엑셀
MAPPING_XLSX = BASE_DIR / "Mapping.xlsx"


# ==== 센서명 매핑 유틸 ====
def _pick_mapping_column(df_map: pd.DataFrame, line: str, product: str | None):
    """엑셀에서 '라인_제품' 열이 있으면 그걸, 없으면 라인으로 시작하는 열 중 우선순위(N>A>기타)로 선택"""
    preferred = f"{line}_{product}" if product else line
    if preferred in df_map.columns:
        return preferred
    cands = [c for c in df_map.columns if str(c).startswith(line)]
    if not cands:
        return None

    def _prio(c):
        u = str(c).upper()
        if "_N_" in u: return 0
        if "_A_" in u: return 1
        return 2

    return sorted(cands, key=_prio)[0]


def load_sensor_map(line: str, product: str | None) -> dict:
    """엑셀(첫열=센서코드)에서 선택 라인의 한글 센서명 매핑을 딕셔너리로 반환.
       - 없거나 읽기 실패하면 빈 dict 반환(기본 영문 코드 그대로 표기)
    """
    try:
        if not MAPPING_XLSX.exists():
            return {}
        dfm = pd.read_excel(MAPPING_XLSX, engine="openpyxl")
    except Exception:
        return {}

    dfm.columns = [str(c).strip() for c in dfm.columns]
    key_col = dfm.columns[0]  # 첫 열 = 센서코드
    tgt_col = _pick_mapping_column(dfm, line, product)
    if not tgt_col:
        return {}
    tmp = (
        dfm[[key_col, tgt_col]]
        .rename(columns={key_col: "CODE", tgt_col: "NAME"})
        .dropna()
    )
    tmp["CODE"] = tmp["CODE"].astype(str).str.strip()
    tmp["NAME"] = tmp["NAME"].astype(str).str.strip()
    tmp = tmp[tmp["NAME"].str.len() > 0]
    return dict(zip(tmp["CODE"], tmp["NAME"]))


# ========================================
# SPC ↔ 불량(Y_Class) 선행탐지 정합성 계산 함수
# ========================================
def evaluate_spc_vs_defect(df_all: pd.DataFrame,
                           alarm_df: pd.DataFrame,
                           time_col: str = "TIMESTAMP",
                           y_col: str = "Y_Quality",
                           cls_col: str = "Y_Class",
                           lead_minutes: int = 30,
                           use_time: bool = True):
    """
    - use_time=True  : TIMESTAMP(시간) 기준으로 [알람 t, t+lead_minutes] 안에 불량 발생했는지
    - use_time=False : 인덱스(샘플) 기준으로 [t, t+lead_minutes] 를 '샘플 수'로 취급
    반환: dict(요약 KPI) + 매칭 테이블(pd.DataFrame)
    """
    out = {
        "n_alarm": 0, "n_defect": 0,
        "tp": 0, "fp": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0,
        "defect_covered": 0, "defect_coverage": 0.0,
        "median_lead_min": None
    }
    match_rows = []

    if df_all is None or df_all.empty or alarm_df is None or alarm_df.empty:
        return out, pd.DataFrame()

    df = df_all.copy()
    df[cls_col] = pd.to_numeric(df.get(cls_col, 1), errors="coerce")
    df["is_defect"] = df[cls_col].isin([0, 2])

    # 시간/인덱스 축 정리
    if use_time:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        alarm_df = alarm_df.copy()
        alarm_df[time_col] = pd.to_datetime(alarm_df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        alarm_df = alarm_df.dropna(subset=[time_col]).sort_values(time_col)
    else:
        df = df.reset_index(drop=True)
        alarm_df = alarm_df.reset_index(drop=True)
        df[time_col] = df.index
        alarm_df[time_col] = alarm_df.index

    if df.empty or alarm_df.empty:
        return out, pd.DataFrame()

    out["n_alarm"] = int(len(alarm_df))
    out["n_defect"] = int(df["is_defect"].sum())

    # 탐색 창
    lead_delta = pd.Timedelta(minutes=int(lead_minutes)) if use_time else int(lead_minutes)

    # 알람별 불량 매칭
    for _, ar in alarm_df.iterrows():
        t0 = ar[time_col]
        mask = ((df[time_col] >= t0) &
                (df[time_col] <= (t0 + lead_delta if use_time else t0 + lead_delta)))
        win = df.loc[mask]
        win_def = win[win["is_defect"]]

        if len(win_def) > 0:
            first_def = win_def.sort_values(time_col).iloc[0]
            out["tp"] += 1
            lead_min = float((first_def[time_col] - t0).total_seconds() / 60.0) if use_time else float(first_def[time_col] - t0)
            match_rows.append({
                "ALARM_TS": t0, "ALARM_Y": ar.get(y_col, None),
                "MATCH_DEFECT_TS": first_def[time_col],
                "MATCH_DEFECT_CLASS": int(first_def[cls_col]) if pd.notna(first_def[cls_col]) else None,
                "LEAD_MIN": lead_min
            })
        else:
            out["fp"] += 1
            match_rows.append({
                "ALARM_TS": t0, "ALARM_Y": ar.get(y_col, None),
                "MATCH_DEFECT_TS": None, "MATCH_DEFECT_CLASS": None, "LEAD_MIN": None
            })

    # 결함 커버리지
    defect_hits = 0
    if out["n_defect"] > 0:
        if use_time:
            for _, dr in df[df["is_defect"]].iterrows():
                dts = dr[time_col]
                has_alarm = alarm_df[(alarm_df[time_col] >= dts - lead_delta) &
                                     (alarm_df[time_col] <= dts)].shape[0] > 0
                defect_hits += int(has_alarm)
        else:
            for _, dr in df[df["is_defect"]].iterrows():
                dts = int(dr[time_col])
                has_alarm = alarm_df[(alarm_df[time_col] >= dts - lead_delta) &
                                     (alarm_df[time_col] <= dts)].shape[0] > 0
                defect_hits += int(has_alarm)

    out["defect_covered"] = int(defect_hits)
    out["defect_coverage"] = (defect_hits / out["n_defect"]) if out["n_defect"] else 0.0

    # 정밀도/재현율/F1
    tp, fp = out["tp"], out["fp"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / out["n_defect"] if out["n_defect"] else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    out["prec"], out["rec"], out["f1"] = float(prec), float(rec), float(f1)

    # 선행시간 중앙값
    matches = pd.DataFrame(match_rows)
    if not matches.empty and matches["LEAD_MIN"].notna().any():
        out["median_lead_min"] = float(matches["LEAD_MIN"].dropna().median())

    return out, matches


# =============================
# 전역 폰트/스타일/템플릿 설정
# =============================
FONT_STACK = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', Arial, sans-serif"

pio.templates.default = "simple_white"
pio.templates["simple_white"].layout.font.family = FONT_STACK
pio.templates["simple_white"].layout.font.size = 12
pio.templates["simple_white"].layout.font.color = "#ffffff"
pio.templates["simple_white"].layout.paper_bgcolor = "rgba(0,0,0,0)"
pio.templates["simple_white"].layout.plot_bgcolor = "rgba(0,0,0,0)"
pio.templates["simple_white"].layout.xaxis.color = "#ffffff"
pio.templates["simple_white"].layout.yaxis.color = "#ffffff"

# 🎨 다크 HMI 스타일
st.markdown("""
<style>
:root{
  --bg:#0e1424; --panel:#101a30; --muted:#9fb6d6; --text:#fff; --grid:#203150;
  --accent:#25c2ff; --border:#1e2a44; --ok:#17d4b3; --warn:#ffe66d; --bad:#ff6b6b;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif!important;}
.block-container{max-width:1500px;margin:auto;padding-top:.6rem;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1222 0%, #0e1424 100%)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
div[data-baseweb="select"]>div,.stTextInput>div>div>input{background:#0b1326!important;color:#fff!important;border:1px solid var(--border)!important;border-radius:10px!important;}
.statusbar{position:sticky;top:0;z-index:999;background:linear-gradient(90deg,#0f182b 0%,#121e36 100%);border:1px solid var(--border);border-radius:12px;box-shadow:0 6px 16px rgba(0,0,0,.28);padding:10px 14px;margin-bottom:12px;}
.status-items{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
.chip{display:inline-flex;align-items:center;gap:8px;background:#0b1326;border:1px solid var(--border);border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700;}
.led{width:10px;height:10px;border-radius:50%;display:inline-block;box-shadow:0 0 12px rgba(255,255,255,.25);}
.led.ok{background:var(--ok);} .led.warn{background:var(--warn);} .led.bad{background:var(--bad);} .led.idle{background:#7e8aa7;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:12px 14px;}
.panel-title{font-size:18px;font-weight:800;margin-bottom:8px;display:flex;align-items:center;gap:8px;}
.panel-title .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px rgba(37,194,255,.6);}
.kpi-container{display:flex;gap:12px;flex-wrap:wrap;}
.kpi-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,.18);flex:1;min-width:240px;padding:18px;text-align:center;}
.kpi-label{font-size:14px;opacity:.9} .kpi-value{font-size:28px;font-weight:800;margin-top:6px;}
.js-plotly-plot .plotly .main-svg{background:transparent!important;}
.panel.glass{background: rgba(255,255,255,0.06) !important;border: 1px solid rgba(255,255,255,0.18) !important;box-shadow: 0 8px 24px rgba(0,0,0,0.25);backdrop-filter: blur(8px) saturate(110%);-webkit-backdrop-filter: blur(8px) saturate(110%);border-radius: 12px;}
.panel.ghost{background: var(--bg) !important;border-color: var(--bg) !important;box-shadow: none !important;}
main [data-testid="stSlider"] { display: none !important; }
.panel:empty { display: none !important; }
main .panel.ghost, main .panel:empty { background: var(--bg) !important; border-color: var(--bg) !important; box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)


# ===== SPC 룰 유틸 =====
def spc_flags(series, mean=None, sigma=None):
    """
    R1: 평균 기준 3σ 밖
    R2: 같은 쪽으로 연속 9점
    R3: 3점 중 2점이 |Z|>2
    """
    x = pd.Series(series).astype(float)
    x = x[pd.notnull(x)]
    m = float(np.nanmean(x)) if mean is None else float(mean)
    s = float(np.nanstd(x, ddof=1)) if sigma is None else float(sigma if sigma != 0 else 1.0)

    flags = pd.DataFrame(index=x.index)
    flags["R1_3sigma"] = (np.abs(x - m) > 3 * s)
    side = np.sign(x - m).replace(0, np.nan).ffill()
    runlen = side.groupby((side != side.shift()).cumsum()).transform("size")
    flags["R2_run9"] = (runlen >= 9)
    z = (x - m) / (s if s != 0 else 1.0)
    r3 = []
    for i in range(len(z)):
        w = z.iloc[max(0, i - 2): i + 1]
        r3.append((np.sum(np.abs(w) > 2) >= 2))
    flags["R3_2of3_over2sigma"] = r3
    flags["ANY"] = flags.any(axis=1)
    return flags, m, s


# -----------------------------
# 데이터 로드 (리포 내 CSV 사용)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"데이터 폴더가 없습니다: {path}\n📦 리포지터리에 'train_ml_imputed/*.csv'를 업로드하세요.")
        st.stop()

    files = sorted(path.glob("*.csv"))
    if not files:
        st.error(f"CSV 파일이 없습니다: {path}\n📦 폴더 안에 학습 CSV들을 업로드하세요.")
        st.stop()

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d["__source__"] = f.name
            dfs.append(d)
        except Exception as e:
            st.warning(f"스킵: {f.name} ({e})")
    data = pd.concat(dfs, ignore_index=True)

    # tz-safe: 우선 naive로 정규화
    if "TIMESTAMP" in data.columns:
        dt = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
    elif "DATE" in data.columns:
        dt = pd.to_datetime(data["DATE"], errors="coerce")
    else:
        st.error("TIMESTAMP/DATE 컬럼이 없습니다."); st.stop()

    data["TIMESTAMP"] = dt
    data["DATE"] = dt.dt.date
    return data


df = load_data(DATA_DIR)

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.markdown("### MENU")
    tab = st.radio("페이지 이동", [" 대시보드", " 이상치 탐지", " 센서 트렌드"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### FILTER")
    dates = ["전체 데이터 기준"] + [str(d) for d in sorted(df["DATE"].dropna().unique())]
    lines = ["전체 라인 기준"] + (sorted(df["LINE"].astype(str).unique()) if "LINE" in df.columns else [])
    sel_date = st.selectbox("DATE", dates, index=0)
    sel_line = st.selectbox("LINE", lines, index=0)

# -----------------------------
# 내부 변수 정의
# -----------------------------
compact = True
hist_h = 300 if compact else 360
pie_h = 280 if compact else 320
rt_h = 320 if compact else 350

# -----------------------------
# Status bar
# -----------------------------
def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

last_ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce").max()
latency = (to_utc(pd.Timestamp.utcnow()) - to_utc(last_ts)).total_seconds() if pd.notnull(last_ts) else None
badge = "ok" if (latency is not None and latency < 300) else ("warn" if (latency is not None and latency < 1800) else "bad")

st.markdown("""
<style>.block-container { padding-top: 2.5rem !important; }</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="statusbar">
  <div class="status-items">
    <span class="chip"><span class="led ok"></span> 연결: Online</span>
    <span class="chip"><span class="led {badge}"></span> 지연: {int(latency) if latency is not None else "-"}s</span>
    <span class="chip"><span class="led idle"></span> 마지막 데이터: {last_ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(last_ts) else "-"}</span>
    <span class="chip"><span class="led ok"></span> 대시보드: Stable</span>
  </div>
</div>
""", unsafe_allow_html=True)


# === 자동 원인 해석 (룰 없어도 동작) ===
def summarize_alarm_cause(df_top5: pd.DataFrame, detected_rule: str | None, sensor_map: dict[str, str] | None = None) -> str:
    if df_top5 is None or df_top5.empty:
        return "원인 후보가 비어 있습니다."

    df = df_top5.copy()
    df["name"] = df["feature"].map(sensor_map).fillna(df["feature"]) if sensor_map else df["feature"]

    rule_desc = {
        "R1_3sigma": "평균 기준 3σ 바깥값 발생 (급격한 이상점)",
        "R2_run9": "같은 방향으로 9점 연속 (드리프트/편향)",
        "R3_2of3_over2sigma": "3점 중 2점이 2σ 초과 (요동/불안정)",
    }
    rule_line = f"감지된 SPC 룰: **{detected_rule}** — {rule_desc.get(detected_rule, '')}" if detected_rule else \
                "감지된 SPC 룰: **없음** (룰 미충족) — 일반 이상 기준으로 해석"

    lines = ["", rule_line, ""]
    for _, r in df.iterrows():
        nm = str(r["name"])
        vr = float(r.get("sigma_ratio", np.nan))
        corr = float(r.get("corr_with_y", np.nan))
        sc = float(r.get("score", np.nan))

        tags = []
        if np.isfinite(vr):
            if vr >= 4:   tags.append("변동성 **매우 큼**")
            elif vr >= 2: tags.append("변동성 **증가**")
        if np.isfinite(corr):
            if abs(corr) >= 0.7: tags.append("Y와 **강한 상관**")
            elif abs(corr) >= 0.4: tags.append("Y와 **중간 상관**")

        tag_txt = " · ".join(tags) if tags else "변동/상관 신호 약함"
        lines.append(f"- **{nm}** → {tag_txt} (score: {sc:.3f})")

    if detected_rule == "R2_run9":
        lines.append("\n> 팁: 추세성(드리프트) 의심 — setpoint 드리프트, 누적 오염, 온도 편차 점검")
    elif detected_rule == "R1_3sigma":
        lines.append("\n> 팁: 급격한 이상점 — 순간 공급 불안정/스파크/급격한 온도 변화 점검")
    elif detected_rule == "R3_2of3_over2sigma":
        lines.append("\n> 팁: 요동 — 제어 루프 튜닝/간헐 노이즈 점검")
    else:
        lines.append("\n> 팁: 상위 후보의 변동성·상관 높은 항목부터 점검")

    return "\n".join(lines)

# -----------------------------
# 대시보드
# -----------------------------
if tab == " 대시보드":
    st.markdown("# 🏭 SMART FACTORY DASHBOARD  ↩︎")
    st.caption("제조라인 실시간 품질 인사이트 및 이상 감지")

    # ── 필터 적용
    dft = df.copy()
    subtitle = "전체 데이터 기준"
    if sel_date != "전체 데이터 기준":
        sd = pd.to_datetime(sel_date).date()
        if "DATE" in dft.columns:
            dft = dft[dft["DATE"] == sd]
        subtitle = f"📅 {sd} 기준"
    if sel_line != "전체 라인 기준" and "LINE" in dft.columns:
        dft = dft[dft["LINE"] == sel_line]
        subtitle += f" | 🏭 {sel_line} 라인 기준"

    # ── KPI (컬럼 부재/공백 안전)
    total = int(len(dft))
    if total and all(c in dft.columns for c in ["Y_Class"]):
        defect = float((dft["Y_Class"] != 1).mean() * 100)
        _bad = dft[dft["Y_Class"] != 1]
    else:
        defect, _bad = 0.0, pd.DataFrame()

    if not _bad.empty and "PRODUCT_CODE" in _bad.columns:
        top_prod = _bad["PRODUCT_CODE"].value_counts().idxmax()
    else:
        top_prod = "-"

    if not _bad.empty and "LINE" in _bad.columns:
        top_line = _bad["LINE"].value_counts().idxmax()
    else:
        top_line = "-"

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-title"><span class="dot"></span>  주요 KPI — '
        f'<span style="color:#9fb6d6">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kpi-container">
          <div class="kpi-card"><div class="kpi-label">📦 총 생산 수량</div><div class="kpi-value">{total:,} EA</div></div>
          <div class="kpi-card"><div class="kpi-label">📉 불량률 (%)</div><div class="kpi-value">{defect:.2f}%</div></div>
          <div class="kpi-card"><div class="kpi-label">🏷️ 주요 불량 제품코드</div><div class="kpi-value">{top_prod}</div></div>
          <div class="kpi-card"><div class="kpi-label">⚙️ 주요 불량 라인</div><div class="kpi-value">{top_line}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 분포/등급 (Y_Quality / Y_Class 없으면 안내)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel glass">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel-title"><span class="dot"></span>  품질 분포 — '
            f'<span style="color:#9fb6d6">{subtitle}</span></div>',
            unsafe_allow_html=True,
        )
        if total and "Y_Quality" in dft.columns:
            fig1 = px.histogram(dft, x="Y_Quality", nbins=25, color_discrete_sequence=["#25c2ff"])
            fig1.update_xaxes(showgrid=True, gridcolor="#203150", zeroline=False)
            fig1.update_yaxes(showgrid=True, gridcolor="#203150", zeroline=False)

            if "Y_Class" in dft.columns and not dft.empty:
                try:
                    m = dft.groupby("Y_Class")["Y_Quality"].mean().sort_index().values
                    if len(m) > 1:
                        for b in [(m[i] + m[i + 1]) / 2 for i in range(len(m) - 1)]:
                            fig1.add_vline(x=b, line=dict(color="#71afff", width=2, dash="dot"))
                except Exception:
                    pass

            fig1.update_layout(
                margin=dict(l=20, r=10, t=10, b=30),
                height=hist_h, paper_bgcolor="#101a30", plot_bgcolor="#101a30",
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Y_Quality 컬럼이 없어 분포 그래프를 표시할 수 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel glass">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel-title"><span class="dot"></span>  품질 등급 비율 — '
            f'<span style="color:#9fb6d6">{subtitle}</span></div>',
            unsafe_allow_html=True,
        )
        if total and "Y_Class" in dft.columns:
            cc = dft["Y_Class"].value_counts(normalize=True).reset_index()
            cc.columns = ["Y_Class", "비율"]
            cc["Y_Class"] = cc["Y_Class"].map({0: "0: 기준 미달", 1: "1: 적합", 2: "2: 기준 초과"}).fillna(cc["Y_Class"].astype(str))
        else:
            cc = pd.DataFrame({"Y_Class": ["데이터 없음"], "비율": [1.0]})

        fig2 = px.pie(
            cc, names="Y_Class", values="비율", hole=0.55,
            color="Y_Class",
            color_discrete_map={
                "0: 기준 미달": "#b3beeb",
                "1: 적합": "#1c2e57",
                "2: 기준 초과": "#6489ee",
                "데이터 없음": "#7e8aa7",
            },
        )
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=pie_h,
                           paper_bgcolor="#101a30", plot_bgcolor="#101a30")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===========================
    # 실시간 품질 추이
    # ===========================
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title" style="font-size:20px; font-weight:600;">'
        '<span class="dot"></span>  실시간 품질 추이 (Y_Quality)'
        "</div>",
        unsafe_allow_html=True,
    )

    # 필수 컬럼 체크
    if "TIMESTAMP" not in dft.columns or "Y_Quality" not in dft.columns:
        st.info("TIMESTAMP 또는 Y_Quality 컬럼이 없어 실시간 추이를 표시할 수 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ─ 필터 변경 감지 → 상태 리셋
    filter_key = f"{sel_date}|{sel_line}"
    if st.session_state.get("active_filter_key") != filter_key:
        st.session_state.active_filter_key = filter_key
        st.session_state.rt_i = 20
        st.session_state.rt_is_playing = True
        st.session_state.alert_log = []
        st.session_state.alert_last_key = None
        st.session_state.alert_c0 = 0
        st.session_state.alert_c2 = 0
        st.session_state.pop("rt_fig", None)

    # 시간 집계(시간당 평균)
    dft["TIMESTAMP"] = pd.to_datetime(dft["TIMESTAMP"], errors="coerce")
    hourly = (
        dft.dropna(subset=["TIMESTAMP"])
        .groupby(dft["TIMESTAMP"].dt.to_period("H"))
        .agg({"Y_Quality": "mean"})
        .reset_index()
    )
    if not hourly.empty:
        hourly["TIMESTAMP"] = hourly["TIMESTAMP"].dt.to_timestamp()
        hourly = hourly.sort_values("TIMESTAMP").reset_index(drop=True)

    # 임계값: 데이터 기반(분위수) → 폴백(고정)
    if len(hourly) >= 10:
        q1, q3 = hourly["Y_Quality"].quantile([0.10, 0.90])
        low_th, high_th = float(q1), float(q3)
    else:
        low_th, high_th = 0.525067, 0.534951  # 폴백

    base_window = 20

    if len(hourly) == 0:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # 실제 윈도우 크기
        window = min(base_window, len(hourly))
        st.session_state.setdefault("rt_is_playing", True)
        st.session_state.setdefault("rt_i", window)
        st.session_state.rt_i = max(window, min(st.session_state.rt_i, len(hourly)))

        # 컨트롤 버튼
        _, ctrl = st.columns([0.75, 0.25])
        with ctrl:
            st.markdown(
                """
                <style>
                div[data-testid="stButton"] button {
                    padding: 0.25rem 0.6rem; height: 1.8rem; font-size: 0.85rem; margin-left: 0.25rem;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            bc1, bc2, bc3 = st.columns(3)

            def _toggle():
                st.session_state.rt_is_playing = not st.session_state.rt_is_playing

            def _step():
                st.session_state.rt_i = min(st.session_state.rt_i + 1, len(hourly))

            def _reset():
                st.session_state.rt_i = window
                st.session_state.alert_log = []
                st.session_state.alert_last_key = None
                st.session_state.alert_c0 = 0
                st.session_state.alert_c2 = 0

            bc1.button("⏯", on_click=_toggle, use_container_width=True)
            bc2.button("⏭", on_click=_step, use_container_width=True)
            bc3.button("⏮", on_click=_reset, use_container_width=True)

        st.session_state.setdefault("rt_speed", 0.6)

        # Plotly Figure
        if "rt_fig" not in st.session_state:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Y_Quality",
                                     line=dict(color="#7fd3ff", width=2)))
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="현재",
                                     marker=dict(size=8, line=dict(color="#7fd3ff", width=1),
                                                 color="rgba(0,0,0,0)")))
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Class 0",
                                     marker=dict(color="#ff6b6b", size=7)))
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Class 2",
                                     marker=dict(color="#ffe66d", size=7)))
            fig.update_layout(
                height=rt_h, margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="#101a30", plot_bgcolor="#101a30",
                xaxis=dict(title="시간", showgrid=False, gridcolor="#203150",
                           zeroline=False, tickformat="%m/%d %H:%M", ticklabelmode="period"),
                yaxis=dict(title="Y_Quality", showgrid=True, gridcolor="#203150",
                           zeroline=False),
                transition={'duration': int(st.session_state.rt_speed*1000), 'easing':'linear'}
            )
            fig.add_hline(y=low_th, line_color="#ff6b6b", line_dash="dot")
            fig.add_hline(y=high_th, line_color="#ffe66d", line_dash="dot")
            st.session_state.rt_fig = fig
        else:
            st.session_state.rt_fig.update_layout(
                height=rt_h,
                transition={'duration': int(st.session_state.rt_speed*1000), 'easing':'linear'}
            )
            st.session_state.rt_fig.update_xaxes(showgrid=False, gridcolor="#203150", zeroline=False)
            st.session_state.rt_fig.update_yaxes(showgrid=True, gridcolor="#203150", zeroline=False)

        # 현재 윈도우 데이터
        i = int(st.session_state.rt_i)
        cur = hourly.iloc[i - window:i]
        if cur.empty:
            st.info("현재 구간 데이터가 없습니다. ‘⏮’을 눌러 재시작하세요.")
        else:
            xs, ys = cur["TIMESTAMP"], cur["Y_Quality"]
            bad = cur[cur["Y_Quality"] < low_th]
            warn = cur[cur["Y_Quality"] > high_th]

            fig = st.session_state.rt_fig
            fig.data[0].x = xs; fig.data[0].y = ys
            last_x = xs.iloc[-1] if len(xs) > 0 else None
            last_y = ys.iloc[-1] if len(ys) > 0 else None
            fig.data[1].x = [last_x] if last_x is not None else []
            fig.data[1].y = [last_y] if last_y is not None else []
            fig.data[2].x = bad["TIMESTAMP"];  fig.data[2].y = bad["Y_Quality"]
            fig.data[3].x = warn["TIMESTAMP"]; fig.data[3].y = warn["Y_Quality"]

            st.plotly_chart(fig, use_container_width=True, key="rt_chart")

            # 실시간 경고 알림(옵션 컬럼 안전)
            def _classify(y, lo, hi):
                if y is None: return 1
                if y < lo: return 0
                if y > hi: return 2
                return 1

            cur_cls = _classify(last_y, low_th, high_th)
            cur_key = (pd.to_datetime(last_x), cur_cls) if last_x is not None else None

            if last_x is not None and cur_cls in (0, 2) and cur_key != st.session_state.get("alert_last_key"):
                # 라인/제품코드가 없을 수 있으니 안전 접근
                nearest_idx = (dft["TIMESTAMP"] - last_x).abs().argsort()[:1]
                line_str = str(dft.iloc[nearest_idx]["LINE"].iloc[0]) if "LINE" in dft.columns else "-"
                if "PRODUCT_CODE" in dft.columns:
                    prod_str = str(dft.iloc[nearest_idx]["PRODUCT_CODE"].iloc[0])
                elif "PRODUCT" in dft.columns:
                    prod_str = str(dft.iloc[nearest_idx]["PRODUCT"].iloc[0])
                else:
                    prod_str = "-"

                st.session_state.alert_log.append({
                    "TIMESTAMP": pd.to_datetime(last_x),
                    "LINE": line_str,
                    "PRODUCT_CODE": prod_str,
                    "Y_Quality": float(last_y) if last_y is not None else None,
                    "CLS": cur_cls
                })
                st.session_state.alert_c0 = st.session_state.get("alert_c0", 0) + int(cur_cls == 0)
                st.session_state.alert_c2 = st.session_state.get("alert_c2", 0) + int(cur_cls == 2)
                st.session_state.alert_last_key = cur_key

                import time  # 파일 상단 임포트에 추가

                # ==================== TOAST STACK (만료시간 기반) ====================
                # 유지 시간(초)
                TOAST_DURATION_SEC = 100000000
                
                if "toast_stack" not in st.session_state:
                    # [{"msg": str, "icon": str, "expires": float}, ...]
                    st.session_state["toast_stack"] = []
                
                now = time.time()
                
                # 새 알람 push (네 로직에서 중복 방지 후 도달)
                msg  = f"[경보] {pd.to_datetime(last_x).strftime('%m/%d %H:%M')} • 클래스 {cur_cls} • Y={float(last_y):.6f}"
                icon = "🛑" if cur_cls == 0 else "⚠️"
                
                # 스택에 추가(최대 3개 유지)
                st.session_state["toast_stack"].append({"msg": msg, "icon": icon, "expires": now + TOAST_DURATION_SEC})
                st.session_state["toast_stack"] = st.session_state["toast_stack"][-3:]
                
                # 만료되지 않은 토스트만 보여주기 (매 rerun마다 다시 띄워져서 오래 보임)
                alive = []
                for t in st.session_state["toast_stack"]:
                    if t["expires"] > now:
                        st.toast(t["msg"], icon=t["icon"])
                        alive.append(t)
                
                # 만료된 항목 정리
                st.session_state["toast_stack"] = alive
            # ── 알람 KPI + 테이블
            st.markdown("#### 🔔 실시간 경고 알림")
            kcol, tcol = st.columns([0.36, 0.64])
            with kcol:
                st.markdown(
                    f"""
                    <div style="display:flex;gap:12px;flex-wrap:wrap;">
                      <div class="kpi-card" style="min-width:200px;border-left:6px solid var(--bad);">
                        <div class="kpi-label">Class 0 (기준 미달)</div>
                        <div class="kpi-value">{st.session_state.get("alert_c0",0)}</div>
                      </div>
                      <div class="kpi-card" style="min-width:200px;border-left:6px solid var(--warn);">
                        <div class="kpi-label">Class 2 (기준 초과)</div>
                        <div class="kpi-value">{st.session_state.get("alert_c2",0)}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with tcol:
                if st.session_state.get("alert_log"):
                    log_df = pd.DataFrame(st.session_state.alert_log)
                    show_cols = [c for c in ["TIMESTAMP", "LINE", "PRODUCT_CODE", "Y_Quality", "CLS"] if c in log_df.columns]
                    st.dataframe(
                        log_df.sort_values("TIMESTAMP", ascending=False)[show_cols].rename(columns={"CLS": "결함클래스"}),
                        use_container_width=True, height=220 if compact else 260,
                    )
                else:
                    st.caption("현재 임계 초과/미달 알람이 없습니다.")

            # 자동 진행
            if st.session_state.rt_is_playing and st.session_state.rt_i < len(hourly):
                time.sleep(float(st.session_state.rt_speed))
                st.session_state.rt_i += 1
                st.rerun()
            elif st.session_state.rt_i >= len(hourly):
                st.caption("")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 이상치 탐지 탭 (경로 자동탐색 + 파일 유연매칭 적용 최종본)
# -----------------------------
elif tab == " 이상치 탐지":
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    # from streamlit_plotly_events import plotly_events  # 필요시 사용

    st.markdown("# 🖥️ Anomaly Detection  ↩︎")
    st.caption("공정 내 주요 변수의 이상 변동 탐지, 이상치 패턴 모니터링")

    # ========= (NEW) 데이터 루트 자동 탐색 =========
    def _pick_data_root() -> Path:
        here = Path(__file__).resolve().parent
        cws  = Path.cwd().resolve()
        cand = [here, cws, here.parent]  # app.py 옆/현재 작업폴더/상위 폴더
        for base in cand:
            if (base / "train_ml_imputed").exists() and (base / "test_ml_imputed").exists():
                return base
        return cws

    _BASE = _pick_data_root()
    TRAIN_DIR = (_BASE / "train_ml_imputed").resolve()
    TEST_DIR  = (_BASE / "test_ml_imputed").resolve()
    OOF_DIR   = (_BASE / "yq_cls_out").resolve()

    # ========= (NEW) 파일 유연 매칭 =========
    def _find_train_file(train_dir: Path, sel_line: str, product: str, expected: str) -> Path | None:
        # 1) 정확 매칭
        p = (train_dir / expected)
        if p.exists():
            return p
        # 2) 대소문자/확장자 변형
        cand_names = {
            expected,
            expected.lower(),
            expected.upper(),
            expected.replace(".csv", ".CSV"),
            expected.replace(".CSV", ".csv"),
        }
        for name in cand_names:
            q = train_dir / name
            if q.exists():
                return q
        # 3) 패턴 매칭(이름 약간 달라도)
        pats = [
            f"{sel_line}_{product}_ml_ready*.csv",
            f"{sel_line}_{product}*ready*.csv",
            f"*{sel_line}*{product}*ready*.csv",
        ]
        for pat in pats:
            hits = sorted(train_dir.glob(pat))
            if hits:
                return hits[0]
        return None

    # ---------- 라인/파일 매핑 ----------
    line_map = {
        "T010305": ("T010305_A_31_ml_ready.csv", "A_31"),
        "T010306": ("T010306_A_31_ml_ready.csv", "A_31"),
        "T050304": ("T050304_A_31_ml_ready.csv", "A_31"),
        "T050307": ("T050307_A_31_ml_ready.csv", "A_31"),
        "T100304": ("T100304_N_31_ml_ready.csv", "N_31"),
        "T100306": ("T100306_N_31_ml_ready.csv", "N_31"),
    }

    # (추천 피처 / 조합별 임계값)
    FEATURE_MAP = {
        "T010305_A_31": ["C_004_ma_w6","C_009_ma_w5","C_013_ma_w6","C_014_ma_w6","C_017_ma_w6","C_021_ma_w6","C_028_ma_w4","C_030_ma_w4","C_033_ma_w4","C_034_ma_w4","C_036_ma_w4","C_003_std_w6","C_004_std_w3","C_014_std_w6","C_017_std_w3","C_021_std_w6","C_025_std_w5","C_028_std_w5","C_033_std_w6","C_034_std_w5","C_036_std_w5","C_016_std_w6"],
        "T010306_A_31": ["C_002_ma_w6","C_013_ma_w3","C_028_ma_w6","C_032_ma_w6","C_033_ma_w6","C_034_ma_w6","C_027_ma_w3","C_006_ma_w3","C_011_std_w5","C_017_std_w3","C_024_std_w4","C_016_std_w3","C_034_std_w5","C_027_std_w4","C_006_std_w3"],
        "T050304_A_31": ["C_003_ma_w3","C_008_ma_w5","C_020_ma_w5","C_026_ma_w5","C_028_ma_w5","C_029_ma_w4","C_034_ma_w6","C_011_ma_w3","C_014_ma_w6","C_013_ma_w6","C_001_std_w3","C_003_std_w3","C_007_std_w6","C_008_std_w5","C_011_std_w4","C_013_std_w6","C_017_std_w3","C_020_std_w6","C_026_std_w6","C_027_std_w6","C_031_std_w3","C_034_std_w6","C_014_std_w6"],
        "T050307_A_31": ["C_001_ma_w6","C_003_ma_w4","C_005_ma_w6","C_008_ma_w5","C_009_ma_w4","C_010_ma_w5","C_011_ma_w6","C_013_ma_w4","C_015_ma_w3","C_016_ma_w6","C_022_ma_w6","C_024_ma_w5","C_025_ma_w6","C_026_ma_w3","C_027_ma_w5","C_028_ma_w4","C_030_ma_w6","C_032_ma_w6","C_034_ma_w6","C_035_ma_w6","C_036_ma_w6","C_017_ma_w6","C_018_ma_w6","C_029_ma_w3","C_031_ma_w6","C_001_std_w6","C_003_std_w6","C_004_std_w6","C_005_std_w3","C_009_std_w6","C_010_std_w5","C_011_std_w6","C_013_std_w4","C_018_std_w6","C_024_std_w5","C_026_std_w3","C_027_std_w3","C_030_std_w5","C_032_std_w5","C_035_std_w5","C_036_std_w5","C_029_std_w5","C_031_std_w6","C_025_std_w6"],
        "T100304_N_31": ["C_024_ma_w6","C_026_ma_w3","C_009_ma_w3","C_030_ma_w3","C_031_ma_w3","C_032_ma_w3","C_007_std_w5","C_011_std_w3","C_016_std_w6","C_022_std_w4","C_026_std_w4","C_029_std_w3","C_030_std_w5","C_032_std_w5","C_014_std_w6","C_023_std_w5"],
        "T100306_N_31": ["C_005_ma_w5","C_018_ma_w5","C_020_ma_w6","C_021_ma_w3","C_027_ma_w6","C_029_ma_w6","C_012_ma_w3","C_009_std_w6","C_011_std_w5","C_016_std_w5","C_018_std_w4","C_021_std_w4","C_030_std_w6","C_025_std_w3"],
    }
    THRESH_MAP = {
        "T010305_A_31": {"sigma":3.0, "iqr":1.0},
        "T010306_A_31": {"sigma":2.5, "iqr":2.0},
        "T050304_A_31": {"sigma":2.0, "iqr":2.0},
        "T050307_A_31": {"sigma":2.5, "iqr":1.0},
        "T100304_N_31": {"sigma":3.0, "iqr":2.0},
        "T100306_N_31": {"sigma":2.5, "iqr":1.0},
    }

    # -----------------------------
    # 제목 + LINE 선택(가로 정렬)
    # -----------------------------
    col1, col2 = st.columns([7, 1])
    with col1:
        st.markdown(" ")
    with col2:
        all_lines = sorted(list(line_map.keys()))
        sel_line = st.selectbox(" ", all_lines, index=0, key="ml_line_sel", label_visibility="collapsed")
        train_fname, product = line_map.get(sel_line, (None, None))
        combo_key = f"{sel_line}_{product}" if product else None

    # ── KPI/TEST (있을 때만)
    st.markdown("""
    <style>
    .mini-kpis.stretch{display:flex;gap:14px;align-items:center;overflow-x:auto;padding:6px 2px}
    .mini-chip{flex:1;background:#0b1326;border:1px solid var(--border);border-radius:999px;padding:12px 20px;height:50px;color:#dce8ff;font-weight:600;white-space:nowrap;display:flex;align-items:center;justify-content:center;box-shadow:0 0 8px rgba(37,194,255,.15)}
    .kpi-label{opacity:.85}.kpi-val{margin-left:6px;font-weight:800;font-size:18px}
    .kpi-val.train{color:#25c2ff;text-shadow:0 0 10px rgba(37,194,255,.85),0 0 20px rgba(37,194,255,.35)}
    .kpi-val.test{color:#ffb86b;text-shadow:0 0 10px rgba(255,184,107,.85),0 0 20px rgba(255,184,107,.35)}
    </style>
    """, unsafe_allow_html=True)

    try:
        oof_glob = list(OOF_DIR.glob(f"oof_summary_{sel_line}_{product}*.csv"))
        test_pred_glob = list(OOF_DIR.glob(f"test_pred_{sel_line}_{product}*.csv"))
        if oof_glob and test_pred_glob:
            oof_df  = pd.read_csv(oof_glob[0])
            test_df = pd.read_csv(test_pred_glob[0])
            from sklearn.metrics import f1_score
            y_true   = pd.to_numeric(oof_df["Y_Class"], errors="coerce")
            y_pred   = pd.to_numeric(oof_df["y_direct_clf"], errors="coerce")
            macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
            weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            micro    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
            cnt = test_df["y_direct_clf"].value_counts().reindex([0,1,2], fill_value=0)
            with st.expander("📎 모델 성능 요약", expanded=False):
                st.markdown(
                    "<div class='mini-kpis stretch'>"
                    f"<div class='mini-chip'><span class='kpi-label'>Macro F1:</span><span class='kpi-val train'>{macro:.3f}</span></div>"
                    f"<div class='mini-chip'><span class='kpi-label'>Weighted F1:</span><span class='kpi-val train'>{weighted:.3f}</span></div>"
                    f"<div class='mini-chip'><span class='kpi-label'>Micro F1:</span><span class='kpi-val train'>{micro:.3f}</span></div>"
                    f"<div class='mini-chip'><span class='kpi-label'>Class 0:</span><span class='kpi-val test'>{int(cnt.loc[0])}</span></div>"
                    f"<div class='mini-chip'><span class='kpi-label'>Class 1:</span><span class='kpi-val test'>{int(cnt.loc[1])}</span></div>"
                    f"<div class='mini-chip'><span class='kpi-label'>Class 2:</span><span class='kpi-val test'>{int(cnt.loc[2])}</span></div>"
                    "</div>", unsafe_allow_html=True
                )
        else:
            st.info("OOF/Test 결과가 없어 KPI는 생략합니다.")
    except Exception as e:
        st.warning(f"KPI 오류: {e}")

    # ============= Train/Test Y_Quality 시계열 =============
    st.markdown(" ")
    st.markdown("### 📈 품질(Y_Quality) 추이")

    try:
        # (NEW) 유연 매칭으로 train 파일 찾기
        expected_name = train_fname if train_fname else ""
        train_fp = _find_train_file(TRAIN_DIR, sel_line, product or "", expected_name) if train_fname else None
        test_fp  = (TEST_DIR / f"test_ml_{sel_line}_{product}.csv") if product else None

        if (train_fp is None) or (not train_fp.exists()):
            st.warning(f"Train 데이터가 없습니다.\n- 데이터 루트: {_BASE}\n- 기대 파일: {expected_name}")
            with st.expander("디버그: train_ml_imputed 파일목록", expanded=False):
                try:
                    names = [p.name for p in list(TRAIN_DIR.glob("*"))[:100]]
                    st.code("\n".join(names) if names else "(비었습니다)")
                except Exception as _e:
                    st.write(_e)
            st.stop()

        df_train = pd.read_csv(train_fp)
        df_test  = (pd.read_csv(test_fp) if (test_fp and test_fp.exists()) else pd.DataFrame())

        # --- 안전 변환 ---
        yq_train  = pd.to_numeric(df_train.get("Y_Quality", pd.Series(dtype=float)), errors="coerce").astype(float).to_numpy()
        cls_train = pd.to_numeric(df_train.get("Y_Class",   pd.Series(dtype=float)), errors="coerce").astype("Int64")

        # x축 결정(시간/인덱스)
        if "TIMESTAMP" in df_train.columns:
            xs_train = pd.to_datetime(df_train["TIMESTAMP"], errors="coerce")
            if xs_train.notna().mean() < 0.8 or xs_train.nunique(dropna=True) < max(10, int(len(xs_train)*0.1)):
                x_is_time, xs_train = False, np.arange(len(yq_train))
            else:
                x_is_time = True
        else:
            x_is_time, xs_train = False, np.arange(len(yq_train))
        x_train = pd.to_datetime(xs_train, errors="coerce") if x_is_time else np.arange(len(yq_train))

        # test 이어붙이기
        if not df_test.empty:
            yq_test = pd.to_numeric(df_test.get("Y_Quality", pd.Series(dtype=float)), errors="coerce").astype(float).to_numpy()
            if "TIMESTAMP" in df_test.columns and x_is_time:
                xs_test = pd.to_datetime(df_test["TIMESTAMP"], errors="coerce")
                if xs_test.notna().mean() < 0.8:
                    x_is_time, xs_test = False, np.arange(len(yq_train), len(yq_train) + len(yq_test))
            else:
                xs_test = np.arange(len(yq_train), len(yq_train) + len(yq_test))
        else:
            yq_test, xs_test = None, None

        # ===== SPC 알람 =====
        _center = float(np.nanmean(yq_train))
        _sigma  = float(np.nanstd(yq_train, ddof=1))
        flags_df, _m, _s = spc_flags(yq_train, mean=_center, sigma=_sigma)
        flags_df = flags_df.reindex(range(len(yq_train))).fillna(False)
        alarm_mask = flags_df["ANY"].to_numpy()
        alarm_df = pd.DataFrame({
            "X": x_train,
            "Y_Quality": yq_train,
            "R1_3sigma": flags_df["R1_3sigma"].to_numpy(),
            "R2_run9":  flags_df["R2_run9"].to_numpy(),
            "R3_2of3_over2sigma": flags_df["R3_2of3_over2sigma"].to_numpy(),
        }).iloc[alarm_mask].copy()
        if not alarm_df.empty:
            rule_cols = ["R1_3sigma","R2_run9","R3_2of3_over2sigma"]
            alarm_df["RULES"] = alarm_df[rule_cols].apply(lambda r: ",".join([c for c in rule_cols if r[c]]), axis=1)

        # ── 알람 히스토리 세션 적재
        if "ml_alarm_hist" not in st.session_state:
            st.session_state["ml_alarm_hist"] = pd.DataFrame(columns=["TIMESTAMP","Y_Quality","RULES"])
        if not alarm_df.empty:
            ts = pd.to_datetime(alarm_df["X"], errors="coerce") if x_is_time else pd.Series(pd.NaT, index=alarm_df.index)
            hist_add = pd.DataFrame({"TIMESTAMP": ts, "Y_Quality": alarm_df["Y_Quality"].values, "RULES": alarm_df.get("RULES","")})
            st.session_state["ml_alarm_hist"] = (
                pd.concat([st.session_state["ml_alarm_hist"], hist_add], ignore_index=True)
                .drop_duplicates(subset=["TIMESTAMP","Y_Quality"])
                .sort_values("TIMESTAMP")
            )

        # y축 범위
        valid_train = np.isfinite(yq_train)
        if valid_train.any():
            y_min = float(np.nanmin(yq_train[valid_train])); y_max = float(np.nanmax(yq_train[valid_train]))
        else:
            y_min, y_max = 0.0, 1.0
        pad = max(1e-3, (y_max - y_min) * 0.1)
        y_range = [y_min - pad, y_max + pad]

        # ----- Figure -----
        fig_yq = go.Figure()

        # UCL/LCL (가능할 때만)
        try:
            class_mins = df_train.groupby("Y_Class")["Y_Quality"].min().sort_index()
            class_maxs = df_train.groupby("Y_Class")["Y_Quality"].max().sort_index()
            if len(class_mins) >= 3 and len(class_maxs) >= 3:
                lcl = (class_maxs.iloc[0] + class_mins.iloc[1]) / 2
                ucl = (class_maxs.iloc[1] + class_mins.iloc[2]) / 2
            else:
                lcl = ucl = None
            if lcl is not None:
                fig_yq.add_hline(y=lcl, line_color="#FFBC3E", line_dash="dot", line_width=2.5,
                                 annotation_text=f"LCL={lcl:.4f}", annotation_position="bottom right",
                                 annotation_font_color="#FFBC3E")
            if ucl is not None:
                fig_yq.add_hline(y=ucl, line_color="#FFBC3E", line_dash="dot", line_width=2.5,
                                 annotation_text=f"UCL={ucl:.4f}", annotation_position="top right",
                                 annotation_font_color="#FFB52D")
        except Exception:
            lcl = ucl = None

        # 본선
        fig_yq.add_trace(go.Scatter(
            x=x_train, y=yq_train, mode="lines+markers",
            line=dict(color="#4AC6D9", width=2), marker=dict(size=5),
            name="Train Y_Quality",
            hovertemplate="x=%{x}<br>Y_Quality=%{y:.6f}<extra></extra>"
        ))

        # 불량/알람/동시
        mask_def = (cls_train.isin([0, 2]).fillna(False).to_numpy()
                    if not cls_train.isna().all() else np.zeros(len(yq_train), bool))
        def_x_all = np.array(x_train)[mask_def] if mask_def.any() else np.array([])
        def_y_all = yq_train[mask_def]          if mask_def.any() else np.array([])

        alarm_x_all = alarm_df["X"].to_numpy() if not alarm_df.empty else np.array([])
        alarm_y_all = alarm_df["Y_Quality"].to_numpy() if not alarm_df.empty else np.array([])

        def _to_num(arr, is_time):
            if arr.size == 0: return np.array([], dtype="int64")
            return pd.to_datetime(arr, errors="coerce").astype("int64") if is_time else np.asarray(arr, dtype="int64")

        def_num   = _to_num(def_x_all,   x_is_time)
        alarm_num = _to_num(alarm_x_all, x_is_time)
        tol = np.int64(15*60*1e9) if x_is_time else np.int64(1)

        both_mask_def   = np.zeros(len(def_num), dtype=bool)
        both_mask_alarm = np.zeros(len(alarm_num), dtype=bool)
        i = j = 0
        while i < len(def_num) and j < len(alarm_num):
            diff = alarm_num[j] - def_num[i]
            if abs(diff) <= tol: both_mask_def[i]=True; both_mask_alarm[j]=True; i+=1; j+=1
            elif diff < 0: j += 1
            else: i += 1

        def_only_x = def_x_all[~both_mask_def] if def_x_all.size else []
        def_only_y = def_y_all[~both_mask_def] if def_y_all.size else []
        both_x     = def_x_all[both_mask_def]  if def_x_all.size else []
        both_y     = def_y_all[both_mask_def]  if def_x_all.size else []
        alarm_only_x = alarm_x_all[~both_mask_alarm] if alarm_x_all.size else []
        alarm_only_y = alarm_y_all[~both_mask_alarm] if alarm_y_all.size else []

        if len(def_only_x):
            fig_yq.add_trace(go.Scatter(x=def_only_x, y=def_only_y, mode="markers",
                                        marker=dict(color="#FA2C7B", size=9), name="불량(Class 0·2)"))
        if len(alarm_only_x):
            fig_yq.add_trace(go.Scatter(x=alarm_only_x, y=alarm_only_y, mode="markers",
                                        marker=dict(color="#FF6FD4", size=9, symbol="diamond"),
                                        name="SPC Alarm"))
        if len(both_x):
            fig_yq.add_trace(go.Scatter(x=both_x, y=both_y, mode="markers",
                                        marker=dict(color="#FFD166", size=11, symbol="star"),
                                        name="이상치·불량 동시발생"))
        if yq_test is not None and xs_test is not None:
            fig_yq.add_trace(go.Scatter(x=xs_test, y=yq_test, mode="lines",
                                        line=dict(color="#FFD700", width=2, dash="dot"),
                                        name="Test 예측 Y_Quality"))

        # 축/레이아웃
        if x_is_time:
            fig_yq.update_xaxes(showgrid=True, gridcolor="#2b3b59",
                                tickformat="%m/%d\n%H:%M", rangeslider=dict(visible=False), zeroline=False)
            x_title = "시간"
        else:
            npts = len(yq_train) + (len(yq_test) if yq_test is not None else 0)
            fig_yq.update_xaxes(showgrid=True, gridcolor="#2b3b59",
                                tickmode="linear", dtick=max(1, int(npts/10)),
                                rangeslider=dict(visible=False), zeroline=False)
            x_title = "시점(Index)"
        fig_yq.update_yaxes(showgrid=True, gridcolor="#2b3b59", range=y_range, zeroline=False)
        fig_yq.update_layout(
            height=320, template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#101a30", plot_bgcolor="#101a30",
            xaxis_title=x_title, yaxis_title="Y_Quality",
            font=dict(color="white", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_yq, use_container_width=True)

        # ================== 이상치 탐지 모델 결과 (정상기반 임계값 + 조합별 적용) ==================
        st.markdown(" ")
        col1, col2 = st.columns([7, 1])
        with col1: st.markdown("### ⚙️ 이상치 탐지 모델 결과")
        with col2:
            all_feats = [c for c in df_train.columns if str(c).startswith("C_")]
            feat_reco = set(FEATURE_MAP.get(combo_key, []))
            disp = [f"{c} ⭐" if c in feat_reco else c for c in all_feats]
            sel_display = st.selectbox("📈 센서 선택", disp, index=0, label_visibility="collapsed",
                                       key=f"feat_all_{combo_key}")
            sel_feature = sel_display.replace(" ⭐", "")

        if sel_feature in df_train.columns:
            y_feat  = pd.to_numeric(df_train[sel_feature], errors="coerce").astype(float)
            xs_time = (pd.to_datetime(df_train["TIMESTAMP"], errors="coerce")
                       if "TIMESTAMP" in df_train.columns else np.arange(len(y_feat)))

            normal_mask = pd.to_numeric(df_train.get("Y_Class", 1), errors="coerce").fillna(1).astype(int).eq(1)
            s_norm = y_feat[normal_mask].dropna().to_numpy()

            _th = THRESH_MAP.get(combo_key, {"sigma":3.0, "iqr":1.5})
            sigma_k, iqr_k = float(_th["sigma"]), float(_th["iqr"])

            from scipy.stats import shapiro
            try:
                stat, p = shapiro(s_norm[:5000]); is_normal = p > 0.05
            except Exception:
                is_normal = False

            if is_normal and len(s_norm) > 3:
                mu, sd = float(np.mean(s_norm)), float(np.std(s_norm))
                lower, upper = mu - sigma_k*sd, mu + sigma_k*sd
                rule_name = f"정상기준 σ({sigma_k})"
            else:
                if len(s_norm) >= 2:
                    q1, q3 = np.percentile(s_norm, [25, 75]); iqr = q3 - q1
                else:
                    q1 = q3 = iqr = np.nan
                if not np.isfinite(iqr) or iqr == 0:
                    lower = np.nanmin(s_norm) if s_norm.size else np.nan
                    upper = np.nanmax(s_norm) if s_norm.size else np.nan
                else:
                    lower, upper = q1 - iqr_k*iqr, q3 + iqr_k*iqr
                rule_name = f"정상기준 IQR({iqr_k})"

            outlier_mask = (y_feat < lower) | (y_feat > upper)

            fig_feat = go.Figure()
            fig_feat.add_trace(go.Scatter(
                x=xs_time, y=y_feat, mode="lines+markers",
                line=dict(color="#4AC6D9", width=2),
                name=sel_feature,
                hovertemplate="시간=%{x}<br>값=%{y:.4f}<extra></extra>"
            ))
            if outlier_mask.any():
                fig_feat.add_trace(go.Scatter(
                    x=np.array(xs_time)[outlier_mask], y=y_feat[outlier_mask],
                    mode="markers", marker=dict(color="#FF6FD4", size=9, symbol="diamond"),
                    name="이상치(모델 임계값)"
                ))

            mask_defect = pd.to_numeric(df_train.get("Y_Class", 1), errors="coerce").isin([0,2]).to_numpy()
            if mask_defect.any():
                fig_feat.add_trace(go.Scatter(
                    x=np.array(xs_time)[mask_defect], y=y_feat[mask_defect],
                    mode="markers", marker=dict(color="#FF002B", size=9),
                    name="불량(Class 0·2)"
                ))
            both = mask_defect & outlier_mask.to_numpy()
            if both.any():
                fig_feat.add_trace(go.Scatter(
                    x=np.array(xs_time)[both], y=y_feat[both],
                    mode="markers", marker=dict(color="#FFD700", size=10, symbol="star"),
                    name="이상치·불량 동시발생"
                ))

            if np.isfinite(lower):
                fig_feat.add_hline(y=lower, line_color="#FFB52D", line_dash="dot",
                                   annotation_text=f"LCL={lower:.4f}", annotation_font_color="#FFB52D")
            if np.isfinite(upper):
                fig_feat.add_hline(y=upper, line_color="#FFB52D", line_dash="dot",
                                   annotation_text=f"UCL={upper:.4f}", annotation_font_color="#FFB52D")

            fig_feat.update_layout(
                height=300, template="plotly_dark",
                margin=dict(l=30, r=30, t=40, b=30),
                title=f"           ↪   {sel_feature} 이상치 탐지 (조합 {combo_key}, {rule_name})",
                xaxis_title="시간", yaxis_title=sel_feature,
                legend=dict(orientation="h", yanchor="bottom", y=1.05,
                            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
            )
            fig_feat.update_xaxes(showgrid=True, gridcolor="#2b3b59", zeroline=False, tickformat="%m-%d")
            st.plotly_chart(fig_feat, use_container_width=True)

        # ===== SPC ALARM =====
        st.markdown("### 🧭 SPC ALARM")
        if "ml_alarm_hist" not in st.session_state or st.session_state["ml_alarm_hist"].empty:
            st.info("현재 SPC 룰 위반 알람이 없습니다.")
        else:
            hist = st.session_state["ml_alarm_hist"].copy()
            sel = st.selectbox(
                "알람 시점",
                options=list(hist["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M:%S")),
                index=len(hist)-1,
                key=f"ml_alarm_pick_{sel_line}"
            )
            pick_ts = pd.to_datetime(sel)
            row = hist.loc[hist["TIMESTAMP"]==pick_ts].iloc[0]
            _ucl_txt = f"{ucl:.4f}" if 'ucl' in locals() and ucl is not None else "-"
            _lcl_txt = f"{lcl:.4f}" if 'lcl' in locals() and lcl is not None else "-"
            st.write(f"- **Y**: `{row['Y_Quality']:.6f}`  | **UCL/LCL**: `{_ucl_txt}` / `{_lcl_txt}`")
            if "RULES" in row and isinstance(row["RULES"], str):
                st.write(f"- **위반 룰**: `{row['RULES']}`")

            try:
                exclude = {"TIMESTAMP","DATE","Y_Quality","Y_Class","__source__"}
                num_cols = [c for c in df_train.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_train[c])]
                if len(num_cols) > 0:
                    # 알람 시점 근처 인덱스
                    if "TIMESTAMP" in df_train.columns and isinstance(pick_ts, pd.Timestamp):
                        ts_all = pd.to_datetime(df_train["TIMESTAMP"], errors="coerce")
                        idx0 = int((ts_all - pick_ts).abs().idxmin())
                    else:
                        idx0 = len(df_train) // 2

                    K_BEFORE, K_AFTER, BASE_SIZE = 5, 5, 20
                    n = len(df_train)
                    l = max(0, idx0 - K_BEFORE); r = min(n - 1, idx0 + K_AFTER)
                    win_df  = df_train.iloc[l:r+1].copy()
                    base_r  = max(0, l - 1)
                    base_l  = max(0, base_r - BASE_SIZE + 1)
                    base_df = df_train.iloc[base_l:base_r+1].copy()

                    out = []
                    y_win = pd.to_numeric(win_df["Y_Quality"], errors="coerce")
                    for c in num_cols[:400]:
                        s_now  = pd.to_numeric(win_df[c],  errors="coerce")
                        s_base = pd.to_numeric(base_df[c], errors="coerce")
                        sigma_now  = float(s_now.std(ddof=1))
                        sigma_base = float(s_base.std(ddof=1)) or 1e-6
                        var_ratio  = sigma_now / sigma_base if np.isfinite(sigma_now) else np.nan
                        corr = s_now.corr(y_win) if s_now.notna().sum() > 5 and y_win.notna().sum() > 5 else np.nan
                        mean_jump = abs(s_now.mean() - s_base.mean())
                        out.append((c, var_ratio, corr, mean_jump))

                    df_top5 = pd.DataFrame(out, columns=["feature","sigma_ratio","corr_with_y","mean_jump"]).dropna(how="all")
                    if not df_top5.empty:
                        df_top5["score1"] = df_top5["sigma_ratio"].clip(0, 5).fillna(0) * df_top5["corr_with_y"].abs().fillna(0)
                        mj_min, mj_max = df_top5["mean_jump"].min(), df_top5["mean_jump"].max()
                        df_top5["score_fb"] = (df_top5["mean_jump"] - mj_min) / (mj_max - mj_min + 1e-9)
                        df_top5["score"] = np.where(df_top5["score1"] > 0, df_top5["score1"], df_top5["score_fb"])
                        df_top5 = df_top5.sort_values("score", ascending=False).head(5)

                    st.caption(f"index window: [{l}:{r}] (win={len(win_df)}), base: [{base_l}:{base_r}] (base={len(base_df)})")

                    sensor_map = load_sensor_map(sel_line, product) if 'load_sensor_map' in globals() else {}
                    if not df_top5.empty and sensor_map:
                        df_top5 = df_top5.copy()
                        df_top5["feature_name"] = df_top5["feature"].map(sensor_map).fillna(df_top5["feature"])
                        show_cols = ["feature","feature_name","sigma_ratio","corr_with_y","score"]
                    else:
                        show_cols = ["feature","sigma_ratio","corr_with_y","score"]

                    if df_top5.empty:
                        st.info("해당 창에서 원인 후보가 비었습니다. K_BEFORE/K_AFTER 또는 BASE_SIZE를 키워보세요.")
                    else:
                        st.dataframe(df_top5[show_cols], use_container_width=True, height=220)
                        detected_rule = None
                        if isinstance(row, pd.Series) and isinstance(row.get("RULES"), str) and len(row["RULES"]) > 0:
                            detected_rule = row["RULES"].split(",")[0].strip()
                        st.markdown("### ⚠️ SPC REPORT")
                        st.info(summarize_alarm_cause(df_top5[["feature","sigma_ratio","corr_with_y","score"]], detected_rule, sensor_map))
                else:
                    st.caption("원인 후보 계산을 위한 수치형 센서 컬럼이 충분하지 않습니다.")
            except Exception as e:
                st.caption(f"원인 후보 계산 스킵: {e}")

            with st.expander("알람 히스토리"):
                st.dataframe(hist.sort_values("TIMESTAMP", ascending=False), use_container_width=True, height=220)
                _csv = hist.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV 다운로드", data=_csv, file_name=f"alarm_history_{sel_line}.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Y_Quality 그래프 로드 중 오류: {e}")

# -----------------------------
# 센서 트렌드
# -----------------------------
elif tab == " 센서 트렌드":
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from pathlib import Path
    import re

    st.markdown("# 📟 SENSOR INFO  ↩︎")
    st.caption("선택 라인의 매핑된 센서만 필터링, 원시값·이동평균·이동표준편차 표시")

    # 앱 폴더 기준 (배포 안전)
    _BASE = BASE_DIR if "BASE_DIR" in globals() else Path(__file__).resolve().parent
    _TRAIN = TRAIN_DIR if "TRAIN_DIR" in globals() else _BASE / "train_ml_imputed"
    _MAP_XLSX = mapping_fp if "mapping_fp" in globals() else (_BASE / "Mapping.xlsx")

    # ---------- 라인/파일 매핑 ----------
    line_map = {
        "T010305": ("T010305_A_31_ml_ready.csv", "A_31"),
        "T010306": ("T010306_A_31_ml_ready.csv", "A_31"),
        "T050304": ("T050304_A_31_ml_ready.csv", "A_31"),
        "T050307": ("T050307_A_31_ml_ready.csv", "A_31"),
        "T100304": ("T100304_N_31_ml_ready.csv", "N_31"),
        "T100306": ("T100306_N_31_ml_ready.csv", "N_31"),
    }
    all_lines = sorted(list(line_map.keys()))
    sel_line = st.selectbox(" LINE", all_lines, index=0, key="sv_line_sel")
    train_fname, product = line_map.get(sel_line, (None, None))

    # ---------- 유틸 ----------
    def _norm(s: str) -> str:
        return re.sub(r"[\s_]+", "", str(s)).upper()

    # ---------- 매핑 로드(엑셀: 첫 열=센서코드, 라인별 열=표시이름) ----------
    def load_line_mapping(fp: Path, line: str, product_suffix: str):
        """엑셀에서 선택 라인의 (코드,이름) 목록 반환."""
        if not fp.exists():
            st.error(f"매핑 파일을 찾을 수 없습니다: {fp}")
            return []

        try:
            dfm = pd.read_excel(fp, engine="openpyxl")
        except Exception as e:
            st.error(f"매핑 파일 읽기 오류: {e}")
            return []

        dfm.columns = [str(c).strip() for c in dfm.columns]

        # 키 열(센서코드) 추정
        key_col = next((c for c in ["센서코드", "SensorCode", "sensor_code", "DF"] if c in dfm.columns), dfm.columns[0])

        # 대상 열(라인 우선순위: 라인_제품 > 라인N > 라인A > 기타 라인)
        preferred = f"{line}_{product_suffix}" if product_suffix else line
        if preferred not in dfm.columns:
            cands = [c for c in dfm.columns if str(c).startswith(line)]
            if not cands:
                return []
            def _prio(c):
                u = str(c).upper()
                if "_N_" in u: return 0
                if "_A_" in u: return 1
                return 2
            preferred = sorted(cands, key=_prio)[0]

        tmp = (dfm[[key_col, preferred]]
               .rename(columns={key_col: "CODE", preferred: "NAME"}))
        tmp["CODE"] = tmp["CODE"].astype(str).str.strip()
        tmp["NAME"] = tmp["NAME"].astype(str).str.strip()
        tmp = tmp[tmp["NAME"].str.len() > 0]   # 이름 비어있는 행 제외
        return [(row["CODE"], row["NAME"]) for _, row in tmp.iterrows()]

    code_name_pairs = load_line_mapping(_MAP_XLSX, sel_line, product)
    if not code_name_pairs:
        st.warning("해당 라인에 매핑된 센서가 없습니다(엑셀 확인).")
        st.stop()

    # ---------- 데이터 로드 ----------
    if not train_fname:
        st.warning("선택 라인의 학습 파일명이 매핑되지 않았습니다.")
        st.stop()

    train_fp = _TRAIN / train_fname
    if not train_fp.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {train_fp}")
        st.stop()

    @st.cache_data(show_spinner=True)
    def _read_csv(fp: Path) -> pd.DataFrame:
        return pd.read_csv(fp)

    df = _read_csv(train_fp)

    # ---------- 시간축 ----------
    if "TIMESTAMP" in df.columns:
        xs_all = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        use_time = xs_all.notna().mean() >= 0.8
        if not use_time:
            xs_all = pd.Series(np.arange(len(df)))
    else:
        xs_all = pd.Series(np.arange(len(df)))
        use_time = False

    # ---------- 실제 존재하는 센서만 필터 ----------
    norm_to_original = {_norm(c): c for c in df.columns}
    available = []
    for code, name in code_name_pairs:
        ncode = _norm(code)
        real_col = None
        if ncode in norm_to_original:
            real_col = norm_to_original[ncode]
        else:
            # 파생열 허용(예: C_002_ma_w3)
            for nc, orig in norm_to_original.items():
                if nc.startswith(ncode):
                    real_col = orig
                    break
        if real_col:
            label = f"{code} · {name}"
            available.append((code, name, real_col, label))

    if not available:
        st.error("매핑된 센서가 데이터 컬럼에 존재하지 않습니다.")
        st.stop()

    # ---------- 멀티 선택 (한 그래프에 겹쳐 그리기) ----------
    labels = [t[3] for t in available]
    # 기본은 0개(깔끔). 필요하면 아래 숫자를 1~3 정도로 바꿔 미리 선택.
    default_labels = [t[3] for t in available[:0]]
    selected_labels = st.multiselect("SENSOR", labels, default=default_labels, key="sv_sel_multi")
    if len(selected_labels) > 5:
        st.warning("가독성을 위해 최대 5개만 권장합니다. 앞의 5개만 표시합니다.")
        selected_labels = selected_labels[:5]
    selected = [t for t in available if t[3] in selected_labels]
    if not selected:
        st.info("표시할 센서를 선택하세요.")
        st.stop()

    # ---------- 한 그래프에 값·MA·SD 겹쳐서 표시 ----------
    colors = ["#00E5FF", "#FF4081", "#F8D210", "#2ECC71", "#9B59B6"]

    fig = go.Figure()
    for i, (code, name, col, label) in enumerate(selected):
        color = colors[i % len(colors)]
        y  = pd.to_numeric(df[col], errors="coerce")
        ma = y.rolling(window=5, min_periods=1).mean()
        sd = y.rolling(window=5, min_periods=1).std()

        fig.add_trace(go.Scatter(x=xs_all, y=y,  mode="lines",
                                 name=label, line=dict(width=2, color=color)))
        fig.add_trace(go.Scatter(x=xs_all, y=ma, mode="lines",
                                 name=f"{label} · MA(5)",
                                 line=dict(width=2, dash="dot", color=color), opacity=0.8))
        fig.add_trace(go.Scatter(x=xs_all, y=sd, mode="lines",
                                 name=f"{label} · SD(5)",
                                 line=dict(width=2, dash="dash", color=color), opacity=0.6))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=10, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="#101a30", plot_bgcolor="#101a30",
        font=dict(color="#FFFFFF"),
        title_text="", legend_title_text=""
    )
    if use_time:
        fig.update_xaxes(showgrid=True, gridcolor="#2b3b59",
                         tickformat="%m/%d\n%H:%M", title_text="")
    else:
        fig.update_xaxes(showgrid=True, gridcolor="#2b3b59",
                         tickmode="linear", title_text="")
    fig.update_yaxes(showgrid=True, gridcolor="#2b3b59", title_text="")

    st.plotly_chart(fig, use_container_width=True)
    st.caption("표시 센서: " + ", ".join([f"{c}({n})" for c, n, _, _ in selected]))

# -----------------------------
# Footer
# -----------------------------
st.caption("© Smart Factory Dashboard — · build time: " +
           datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


















