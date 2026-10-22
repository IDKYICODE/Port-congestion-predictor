"""Streamlit demo: pick a real 2024 port call, see whether it will wait > 24h for a berth, why, and what to do."""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent / "src"))
from explain import CAUSES, apply_scenario, displaced, explain_row, free_berth_norms, load_monthly, whatif_predict, queue_caps, situation, terminal_norms  # noqa: E402
from load import PROC  # noqa: E402
from port_outlook import outlook  # noqa: E402
from scene import scene_data, scene_html  # noqa: E402

st.set_page_config(page_title="Port Delay Control Room", page_icon="⚓", layout="wide")

RED, AMBER, GREEN = "#f0716b", "#f2b04a", "#4cc38a"
SHORT = {  # cause label -> phrase used inside a sentence
    CAUSES["line"]: "a line of ships at the terminal", CAUSES["berths"]: "berths are busy",
    CAUSES["slow"]: "the terminal has been slow lately", CAUSES["port"]: "the whole port is congested",
    CAUSES["ship"]: "this ship's own track record", CAUSES["cargo"]: "its cargo type and size",
    CAUSES["weather"]: "the weather", CAUSES["calendar"]: "the time of year", CAUSES["identity"]: "this terminal's usual pattern",
}

st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}}
h1 {{font-size: 1.55rem !important; letter-spacing: -0.02em; margin-bottom: .1rem !important;}}
.sub {{color: #9db3d1; font-size: .92rem; margin-bottom: 1.4rem;}}
.verdict {{display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end;
           padding: 22px 26px; border-radius: 14px; background: #10233d; margin-bottom: 18px;}}
.verdict .head {{font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15;}}
.verdict .line {{color: #c9d8ee; font-size: 1.02rem; margin-top: 6px; max-width: 70ch;}}
.verdict .why {{color: #9db3d1; font-size: .95rem; margin-top: 10px; max-width: 75ch;}}
.verdict .side {{text-align: right; white-space: nowrap;}}
.verdict .big {{font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;}}
.verdict .small {{color: #9db3d1; font-size: .9rem;}}
.tag {{display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 99px; font-size: .82rem;
       background: #17304f; color: #c9d8ee;}}
.dot {{display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 1px;}}
.panel h4 {{font-size: 1rem; margin: 0 0 10px; color: #e6eefb;}}
.bars {{margin-bottom: 18px;}}
.bar {{display: grid; grid-template-columns: 1fr 64px; gap: 10px; align-items: center; margin: 7px 0;}}
.bar .name {{font-size: .88rem; color: #c9d8ee; margin-bottom: 3px;}}
.bar .track {{position: relative; height: 8px; border-radius: 4px; background: #17304f;}}
.bar .fill {{position: absolute; top: 0; height: 8px; border-radius: 4px;}}
.bar .mid {{position: absolute; left: 50%; top: -3px; width: 1px; height: 14px; background: #3a5a82;}}
.bar .val {{font-size: .88rem; font-variant-numeric: tabular-nums; text-align: right;}}
.legend {{color: #8ea6c6; font-size: .78rem; display: flex; justify-content: space-between; margin-top: 2px;}}
.todo {{margin: 0; padding-left: 18px; color: #dbe6f6; font-size: .93rem;}} .todo li {{margin: 6px 0;}}
.unusual {{display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 6px 0 18px;}}
.fact {{background: #10233d; border-radius: 12px; padding: 12px 14px;}}
.fact .k {{color: #9db3d1; font-size: .82rem;}}
.fact .v {{font-size: 1.15rem; font-weight: 650; margin-top: 2px;}}
.fact .n {{color: #8ea6c6; font-size: .8rem;}}
.gauge {{max-width: 460px; margin: 12px 0 6px;}}
.g-track {{position: relative; height: 8px; border-radius: 4px;}}
.g-mark {{position: absolute; top: -5px; width: 4px; height: 18px; margin-left: -2px; border-radius: 2px;
          background: #ffffff; box-shadow: 0 0 0 2px #10233d;}}
.g-alarm {{position: absolute; top: -3px; width: 1px; height: 14px; background: #e6eefb; opacity: .7;}}
.g-labels {{position: relative; height: 16px; font-size: .74rem; color: #8ea6c6; margin-top: 4px;}}
.g-labels span {{position: absolute; transform: translateX(-50%); white-space: nowrap;}}
.g-labels span:first-child {{left: 0; transform: none;}} .g-labels span:last-child {{right: 0; transform: none;}}
.compare {{display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline; margin-top: 10px; padding: 8px 12px;
           border-radius: 10px; background: #17304f; color: #c9d8ee; font-size: .92rem; max-width: 560px;}}
.compare .arrow {{color: #8ea6c6;}} .compare .delta {{color: #f2b04a; font-variant-numeric: tabular-nums;}}
@media (max-width: 760px) {{ .verdict {{grid-template-columns: 1fr;}} .verdict .side {{text-align: left;}} }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model for this month…")
def bundle(h, month):
    return load_monthly(h, month)


@st.cache_data(show_spinner="Loading 2020–2024 port records…")
def features(h):
    return pd.read_parquet(PROC / f"features_h{h}.parquet")


@st.cache_data(show_spinner=False)
def free_norms(h):
    return free_berth_norms(features(h))


@st.cache_data(show_spinner=False)
def caps(h):
    return queue_caps(features(h))


@st.cache_data(show_spinner=False)
def preds(h):
    return pd.read_parquet(PROC / f"preds_2024_h{h}.parquet")


@st.cache_data(show_spinner=False)
def norms(h):
    return terminal_norms(features(h))


@st.cache_data(show_spinner=False)
def confidence_check(h):
    import json
    return json.loads((Path(__file__).parent / "report" / "confidence_port_check.json").read_text(encoding="utf-8"))[f"h{h}"]


@st.cache_data(show_spinner=False)
def test_metrics(h):
    import json
    return json.loads((Path(__file__).parent / "report" / f"metrics_monthly_h{h}.json").read_text(encoding="utf-8"))


def hours(x):
    return f"{x / 24:.1f} days" if x >= 48 else f"{x:.0f}h"


def show_html(src, height):
    """3D scene (our own generated HTML). st.iframe replaces components.html, which newer Streamlit drops."""
    if hasattr(st, "iframe"):
        st.iframe(src, height=height)
    else:
        components.html(src, height=height)


def flat(s):
    """One line per HTML tag, no indentation or blank lines: Markdown would turn an indented line after a blank one into a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


# sidebar: choose a ship, then optional what-if
with st.sidebar:
    st.markdown("### Choose a ship")
    when = st.radio("Predict", ["At arrival", "3 days before"], horizontal=True,
                    help="'3 days before' is less accurate but leaves time to slow down or divert.")
    h = 0 if when == "At arrival" else 72
    rows = features(h)
    rows = rows[rows.arrival >= "2024-01-01"]
    ports = sorted(rows.port.astype(str).unique())
    qp = st.query_params
    if qp.get("port") in ports and "port" not in st.session_state:
        st.session_state["port"] = qp["port"]
        if qp.get("ship", "").isdigit():
            st.session_state["ship"] = int(qp["ship"])
    port = st.selectbox("Port", ports, index=None, placeholder="Choose a port", key="port")
if port is None:
    st.title("Port Delay Control Room")
    st.markdown('<div class="sub">Will a ship wait more than 24 hours for a berth, and why? '
                'Uses real 2024 arrivals at 15 Brazilian ports.</div>', unsafe_allow_html=True)
    # start screen: every port's outlook for one week, riskiest first; click a row to open that port
    wk = st.date_input("Week starting", value=pd.Timestamp("2024-03-04"), min_value=pd.Timestamp("2024-01-01"),
                       max_value=pd.Timestamp("2024-12-25"), key="week", format="DD/MM/YYYY")
    pr = preds(h)
    table = [dict(o, port=p) for p in sorted(pr.port.unique()) if (o := outlook(pr, p, wk))]
    ov = pd.DataFrame({
        "Port": [o["port"] for o in table],
        "Ships arriving": [o["ships"] for o in table],
        "Likely delayed > 24h": [round(o["share"] * 100) for o in table],
        "Typical wait": [hours(o["typical_wait_h"]) for o in table],
        "Riskiest terminal": [o["busiest_terminal"] for o in table],
        "Really delayed (2024 record)": [o["real_delayed"] for o in table],
    }).sort_values("Likely delayed > 24h", ascending=False, ignore_index=True)
    st.session_state["overview_ports"] = list(ov.Port)

    def open_port():
        sel = st.session_state["overview"].selection.rows
        if sel:
            st.session_state["port"] = st.session_state["overview_ports"][sel[0]]

    st.markdown(f"#### All ports, week of {pd.Timestamp(wk):%d %b %Y}")
    st.caption("Riskiest first. Click a row to open that port, or choose one in the sidebar.")
    st.dataframe(ov, hide_index=True, width="stretch", key="overview", on_select=open_port,
                 selection_mode="single-row",
                 column_config={"Likely delayed > 24h": st.column_config.ProgressColumn(
                     "Likely delayed > 24h", format="%d%%", min_value=0, max_value=100)})
    st.stop()

with st.sidebar:
    sub = rows[rows.port.astype(str) == port].sort_values("arrival")
    if st.session_state.get("ship", 0) >= len(sub):     # list is shorter after switching port
        st.session_state["ship"] = 0
    labels = [f"{r.arrival:%d %b %Y, %H:%M} · {r.terminal} · {r.cargo_nature}" for r in sub.itertuples()]
    pick = st.selectbox("Ship arrival", range(len(sub)), format_func=lambda i: labels[i], key="ship")
    row = sub.iloc[[pick]].copy()

    st.divider()
    st.markdown("### What if…")
    q0 = int(row.term_queue.iloc[0])
    c0 = int(np.nan_to_num(row.term_berths.iloc[0], nan=1))
    term = str(row.terminal.iloc[0])
    qmax = max(q0, int(caps(h).get(term, 15)))
    q = st.slider("Ships waiting at this terminal", 0, qmax, q0,
                  help="Capped near the most ships this terminal really had in line (2020–2022).")
    closed = st.slider(f"Berths closed today (of {c0})", 0, c0 - 1, 0,
                       help="Crane breakdown, repairs, weather. You can only take real berths away; "
                            "the model never saw a terminal gain berths.") if c0 > 1 else 0
    scenario = (q, closed) != (q0, 0)
    real_row = row
    if scenario:
        moved = displaced(row, c0 - closed)
        row = apply_scenario(row, q, c0 - closed)
        st.caption("Scenario active: showing a what-if, not the real situation."
                   + (f" {moved} ship(s) at the closed berths had to leave and join the line." if moved else ""))

month = f"{row.arrival.iloc[0]:%Y-%m}"
b = bundle(h, month)
r = whatif_predict(b, real_row, q, closed, qmax) if scenario else explain_row(b, row)
lo, mid, hi = r["range_h"]
prob, thr = r["prob"], b["thr"]
actual = float(row.wait_h.iloc[0])

# header
st.title("Port Delay Control Room")
arr = row.arrival.iloc[0]
st.markdown(f'<div class="sub">{html.escape(term)} · {html.escape(port)} · arrived {arr:%d %b %Y, %H:%M} · '
            f'{html.escape(str(row.cargo_nature.iloc[0]))}</div>', unsafe_allow_html=True)

# verdict
if prob >= 0.8:
    color, head = RED, "Very likely delayed"
elif prob >= 0.5:
    color, head = RED, "Likely delayed"
elif prob >= thr:
    color, head = AMBER, "Possible delay"
else:
    color, head = GREEN, "Likely on time"
# risk gauge: green below the alarm line, amber up to 50%, red above; the marker is this ship
gauge = (f'<div class="gauge"><div class="g-track" style="background:linear-gradient(90deg,{GREEN} 0 {thr:.0%},'
         f'{AMBER} {thr:.0%} 50%,{RED} 50% 100%)"><div class="g-mark" style="left:{prob:.1%}"></div>'
         f'<div class="g-alarm" style="left:{thr:.0%}"></div></div>'
         f'<div class="g-labels"><span>0%</span><span style="left:{thr:.0%}">alarm line {thr:.0%}</span>'
         f'<span style="left:50%">50%</span><span>100%</span></div></div>')
compare = ""
if scenario:   # the same ship in its real situation, for a before / after
    r0 = explain_row(b, real_row)
    compare = (f'<div class="compare"><span>Real situation <b>{r0["prob"]:.0%}</b> · {hours(r0["range_h"][1])}</span>'
               f'<span class="arrow">→</span><span>This what-if <b style="color:{color}">{prob:.0%}</b> · {hours(mid)}</span>'
               f'<span class="delta">{(prob - r0["prob"]) * 100:+.0f} pts</span></div>')
up = [x for x in r["causes"] if x["effect_pp"] >= 2]   # only causes that really move the risk
if up:
    why = "Mainly because " + ", ".join(SHORT.get(x["cause"], x["cause"].lower()) for x in up[:3]) + "."
else:
    why = "Nothing is pushing the risk above normal for this terminal."
if scenario:
    tag = "What-if scenario · real outcome hidden"
else:
    right = (actual > 24) == (prob >= thr)
    tag = (f"What really happened: waited {hours(actual)} · "
           f"{'prediction was right' if right else 'prediction missed'}")
conf = max(prob, 1 - prob)
if prob >= 0.5:
    conf_note = "that this ship will be delayed"
elif prob >= thr:
    conf_note = "that this ship will be on time (alarm raised anyway as a precaution: a missed delay costs more)"
else:
    conf_note = "that this ship will be on time"
free_note = ""
if float(row.term_queue.iloc[0]) == 0 and float(row.term_occupancy.iloc[0]) < float(row.term_berths.iloc[0]):
    base = free_norms(h).get(term)
    if base is not None and base == base:
        slow = float(np.nan_to_num(row.term_wait_mean_7d.iloc[0])) > 1.5 * float(np.nan_to_num(norms(h).loc[term, "term_wait_mean_7d"] if term in norms(h).index else np.inf))
        free_note = (f'<div class="why">No line and a berth is free. Even then, ships here typically take about {hours(base)} '
                     f'from arrival to berth (channel, pilot, tide, mooring, paperwork)'
                     + ("; this terminal has also been slower than usual this week." if slow else ".") + '</div>')
status_col = {"normal": GREEN, "slightly disrupted": AMBER, "heavily disrupted": RED}[r["status"]]
st.markdown(flat(f"""<div class="verdict">
  <div>
    <div class="head" style="color:{color}">{head}</div>
    <div class="line"><b>{prob:.0%}</b> chance this ship waits more than 24 hours for a berth.</div>
    {gauge}
    {compare}
    <div class="line" style="margin-top:4px"><b>Confidence: {conf:.0%}</b> <span style="color:#9db3d1">{conf_note}</span></div>
    <div class="why">{why}</div>
    {free_note}
    <span class="tag">{tag}</span>
  </div>
  <div class="side">
    <div class="small">Expected wait</div>
    <div class="big">{hours(mid)}</div>
    <div class="small">80% likely between {hours(lo)} and {hours(hi)}</div>
    <div class="small" style="margin-top:8px"><span class="dot" style="background:{status_col}"></span>Port this week: {r["status"]}</div>
  </div>
</div>"""), unsafe_allow_html=True)

# port outlook: the week starting at this arrival (real predictions, not the what-if)
po = outlook(preds(h), port, row.arrival.iloc[0])
if po:
    st.markdown(
        f'<div class="fact" style="margin-bottom:16px"><div class="k">Port outlook · {html.escape(port)} · '
        f'{po["start"]:%d %b} – {po["end"]:%d %b %Y} (all arrivals, real situation)</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:28px;margin-top:6px">'
        f'<div><div class="v">{po["ships"]}</div><div class="n">ships arriving</div></div>'
        f'<div><div class="v" style="color:{RED if po["share"] >= 0.5 else AMBER if po["share"] >= 0.25 else GREEN}">'
        f'~{po["exp_delayed"]:.0f} ({po["share"]:.0%})</div><div class="n">likely delayed &gt; 24h</div></div>'
        f'<div><div class="v">{hours(po["typical_wait_h"])}</div><div class="n">typical expected wait</div></div>'
        f'<div><div class="v" style="font-size:1rem">{html.escape(po["busiest_terminal"])}</div>'
        f'<div class="n">busiest terminal ({po["busiest_risk"]:.0%} delay risk)</div></div>'
        f'<div><div class="v">{po["real_delayed"]}</div><div class="n">really delayed (2024 record)</div></div>'
        f'</div></div>', unsafe_allow_html=True)

# scene + why
show_html(scene_html(scene_data(row, r, total_berths=c0, closed=closed, thr=thr,
                                      whatif={"queue": q, "real_queue": q0, "closed": closed,
                                              "moved": displaced(real_row, c0 - closed)} if scenario else None), height=430), height=440)
st.write("")
left, right_col = st.columns([3, 2], gap="large")
with left:
    shown = [x for x in r["causes"] if abs(x["effect_pp"]) >= 1][:6]
    scale = max([abs(x["effect_pp"]) for x in shown] + [1])
    rows_html = ""
    for x in shown:
        w = abs(x["effect_pp"]) / scale * 50
        side = f"left:50%;width:{w:.1f}%;background:{RED}" if x["effect_pp"] > 0 else f"right:50%;width:{w:.1f}%;background:{GREEN}"
        rows_html += (f'<div class="bar"><div><div class="name">{html.escape(x["cause"])}</div>'
                      f'<div class="track"><div class="mid"></div><div class="fill" style="{side}"></div></div></div>'
                      f'<div class="val" style="color:{RED if x["effect_pp"] > 0 else GREEN}">{x["effect_pp"]:+.0f} pts</div></div>')
    st.markdown(f'<div class="panel"><h4>Why the model thinks so</h4><div class="bars">{rows_html or "No strong causes."}'
                f'<div class="legend"><span>lowers the risk</span><span>raises the risk</span></div></div></div>',
                unsafe_allow_html=True)
with right_col:
    who = "the ship operator (still at sea)" if h else "port and landside planners"
    st.markdown(f'<div class="panel"><h4>What to do</h4><ul class="todo">'
                + "".join(f"<li>{html.escape(a)}</li>" for a in r["actions"])
                + f'</ul><div style="color:#8ea6c6;font-size:.82rem;margin-top:8px">Advice for {who}.</div></div>',
                unsafe_allow_html=True)

# only what's unusual
facts = situation(row, norms(h))
odd = [f for f in facts if f["flag"] != "usual"][:4]
st.markdown("#### What's unusual today")
if odd:
    cards = "".join(
        f'<div class="fact"><div class="k">{html.escape(f["fact"])}</div>'
        f'<div class="v" style="color:{RED if f["flag"] == "worse" else GREEN}">{html.escape(f["now"])}</div>'
        f'<div class="n">normally {html.escape(f["normal"])} here</div></div>' for f in odd)
    st.markdown(f'<div class="unusual">{cards}</div>', unsafe_allow_html=True)
else:
    st.caption("Everything is close to normal for this terminal.")

# details on demand
with st.expander("All facts compared with normal"):
    t = pd.DataFrame(facts).rename(columns={"fact": "Fact", "now": "Now", "normal": "Normal here", "flag": "vs normal"})
    t["vs normal"] = t["vs normal"].map({"worse": "worse", "better": "better", "usual": "usual"})
    st.table(t.set_index("Fact"))
    st.caption("'Normal' = this terminal's median in 2020–2022. All facts are as of the prediction moment.")
with st.expander("Individual reasons (model detail)"):
    for x in r["reasons"]:
        st.markdown(f"**{x['effect_pp']:+.0f} pts** · {x['text']} <span style='color:#8ea6c6'>({x['cause']})</span>",
                    unsafe_allow_html=True)
    st.caption("SHAP values grouped and converted to approximate percentage points.")
with st.expander("How accurate is this?"):
    m = test_metrics(h)
    st.markdown(f"Tested on **all {m['n_test']:,} arrivals of 2024** ({when.lower()}), which the model never saw while it was built.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Yes/no answers right", f"{m['accuracy_at_0.5']:.1%}")
    k2.metric("Real delays caught", f"{m['recall_at_thr']:.1%}")
    k3.metric("Typical hour error", f"{m['hours_median_err']:.1f}h")
    k4.metric("Real wait inside range", f"{m['range_coverage_80']:.0%}")
    cc = pd.DataFrame(confidence_check(h)["confidence"])
    st.markdown("**Can you trust the confidence number?** When the app showed this confidence, this is how often it was right:")
    st.dataframe(pd.DataFrame({
        "Confidence shown": [f"~{c:.0%}" for c in cc.avg_confidence],
        "Actually right": [f"{a:.1%}" for a in cc.actually_right],
        "Ships": [f"{n:,}" for n in cc.ships]}), hide_index=True, width="stretch")
    st.caption(f"Model: LightGBM retrained on {b['trained_until']} with every arrival whose outcome was known by then "
               f"({len(b['features'])} hints). Alarm line {thr:.0%}: above it we flag a delay, because a missed delay is "
               "assumed to cost 3× a false alarm. Data: ANTAQ port calls (Brazil) and Open-Meteo weather.")
