"""Live progress dashboard for the item 7 training queue (localhost only, read-only).

Reads the resumable training state files (results/p1_item7_train_<recipe>_seed<S>.json),
the DS2 evaluation log and nvidia-smi; never writes to results/ or checkpoints/.

Run (from repo root):
    python reports/live_dashboard.py            # http://127.0.0.1:8765
    python reports/live_dashboard.py --recipe r4 --compare r3 --port 8765
"""
import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 202, 303)
TRAIN_WINDOWS = {"r2": None, "r3": 375, "r4": 447}
EVAL_EVERY = {"r3": 93, "r4": 111}
_gpu = {"t": 0.0, "v": None, "hist": []}
_rate = {}  # seed -> list of (time, updates)


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_summary(recipe, seed):
    path = ROOT / f"results/p1_item7_train_{recipe}_seed{seed}.json"
    s = load(path)
    if not s:
        return None
    mtime = path.stat().st_mtime
    evals = [{"update": h["update"], "bal": h["val"]["balanced_accuracy"], "parse": h["val"]["parse_rate"],
              "by_n": h["val"]["by_n"]} for h in s["history"] if "val" in h]
    losses = [(h["update"], h["loss"]) for h in s["history"]]
    k = 25
    smooth = [(losses[i][0], sum(l for _, l in losses[i - k + 1:i + 1]) / k) for i in range(k - 1, len(losses), 5)]
    return {"updates": s["updates"], "done": s["done"], "best_update": s["best_update"],
            "best_val": s.get("best_val"), "evals_since_best": s["evals_since_best"], "evals": evals,
            "loss": smooth, "lr": s["history"][-1].get("lr") if s["history"] else None, "saved_at": mtime}


def gpu():
    if time.time() - _gpu["t"] > 1.5:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            u, m, mt, t, p = [x.strip() for x in out.split(",")]
            _gpu["v"] = {"util": float(u), "mem": float(m), "mem_total": float(mt), "temp": float(t), "power": float(p)}
            _gpu["hist"].append((round(time.time(), 1), float(u), float(p)))
            del _gpu["hist"][:-180]
        except Exception:
            _gpu["v"] = None
        _gpu["t"] = time.time()
    return dict(_gpu["v"], hist=_gpu["hist"]) if _gpu["v"] else None


def eval_progress(recipe):
    log = ROOT / f"logs/p1_item7_eval_ds2v2_{recipe}.log"
    if not log.exists():
        return None
    arms = {}
    for m in re.finditer(r"\[([^\]\s]+) (\d+)/(\d+)\]", log.read_text(encoding="utf-8", errors="replace")):
        arms[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    done = re.findall(r"\[([^\]\s]+)\] done", log.read_text(encoding="utf-8", errors="replace"))
    return {"arms": {a: {"done": d, "total": t} for a, (d, t) in arms.items()}, "finished_arms": done,
            "complete": (ROOT / f"results/p1_item7_eval_ds2v2_{recipe}.json").exists() and "QUEUE" not in ""}


def status(recipe, compare):
    now = time.time()
    total = TRAIN_WINDOWS.get(recipe, 447) * 3
    runs, stages, active = {}, [], None
    for seed in SEEDS:
        r = run_summary(recipe, seed)
        runs[str(seed)] = r
        state = "done" if r and r["done"] else ("running" if r else "waiting")
        if state == "running" and active is None:
            active = seed
            hist = _rate.setdefault(seed, [])
            if not hist or hist[-1][1] != r["updates"]:
                hist.append((r["saved_at"], r["updates"]))  # save time, not poll time
            del hist[:-40]
        elif state == "running":
            state = "paused"
        stages.append({"name": f"{recipe} seed {seed}", "state": state,
                       "progress": (r["updates"] / total) if r else 0})
    ev = eval_progress(recipe)
    ev_done = (ROOT / f"results/p1_item7_r4_analysis.json").exists() if recipe == "r4" else False
    ev_state = "waiting"
    if ev:
        ev_state = "running"
        if ev["finished_arms"] and len(ev["finished_arms"]) >= 4:
            ev_state = "done"
    stages.append({"name": "DS2 v2 test evaluation", "state": ev_state, "detail": ev})
    stages.append({"name": "r4 analysis", "state": "done" if ev_done else "waiting"})
    eta = None
    if active is not None:
        h, ev = _rate.get(active, []), EVAL_EVERY.get(recipe, 111)
        train, val = [], []
        for (t0, u0), (t1, u1) in zip(h, h[1:]):
            if u1 <= u0:
                continue
            if u0 // ev == u1 // ev:          # no validation check in between: pure training speed
                train.append((t1 - t0) / (u1 - u0))
            else:                               # interval contains one check: the excess is validation time
                val.append(t1 - t0)
        spu = sorted(train)[len(train) // 2] if train else 5.0      # prior until measured (seed 303 log: ~16 min per 111 updates incl. a check)
        val_s = max(0.0, (sorted(val)[len(val) // 2] - 25 * spu)) if val else 420.0
        u = runs[str(active)]["updates"]
        n_checks = total // ev - u // ev
        eta = {"sec_per_update": spu, "val_sec": val_s, "measured": bool(train), "val_measured": bool(val),
               "saved_update": u, "saved_at": runs[str(active)]["saved_at"],
               "to_max": (total - u) * spu + n_checks * val_s, "eval_every": ev}
    ref = {str(s): run_summary(compare, s) for s in SEEDS} if compare else {}
    return {"time": time.strftime("%H:%M:%S"), "now": now, "recipe": recipe, "compare": compare, "total_updates": total,
            "min_updates": int(total * 2 / 3), "active_seed": active, "runs": runs, "reference": ref,
            "stages": stages, "gpu": gpu(), "eta": eta}


class Handler(BaseHTTPRequestHandler):
    recipe, compare = "r4", "r3"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(status(self.recipe, self.compare)).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Item 7 Live Progress</title>
<style>
:root{color-scheme:light;--bg:#f4f4f2;--surface:#fcfcfb;--border:#e3e2dd;--grid:#ecebe7;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#008300;--warn:#b36b00;--track:#e8e7e2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111110;--surface:#1a1a19;--border:#2e2e2c;--grid:#262624;--text:#fff;--text2:#c3c2b7;--muted:#8d8c85;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#3fb23f;--warn:#e0a030;--track:#2a2a28}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--border:#2e2e2c;--grid:#262624;--text:#fff;--text2:#c3c2b7;--muted:#8d8c85;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#3fb23f;--warn:#e0a030;--track:#2a2a28}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px;margin-bottom:16px}
h1{font-size:20px;margin:0;font-weight:650}.sub{color:var(--text2)}.live{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--good);margin-right:6px;animation:p 2s infinite}@keyframes p{50%{opacity:.35}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text2);font-size:12px}.tile .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}.tile .n{color:var(--muted);font-size:12px}
.bar{height:6px;background:var(--track);border-radius:3px;margin-top:8px;overflow:hidden}.bar i{display:block;height:100%;background:var(--s1);border-radius:3px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}@media(max-width:820px){.grid2{grid-template-columns:1fr}}
h2{font-size:14px;margin:0 0 2px;font-weight:600}.cap{color:var(--muted);font-size:12px;margin-bottom:8px}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--text2);margin-bottom:6px}.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:3px;border-radius:2px;display:inline-block}.sw.d{background:none!important;border-top:2px dashed}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9;min-width:150px}
.tip b{font-weight:600}.tip .r{display:flex;justify-content:space-between;gap:12px;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}th{color:var(--text2);font-weight:500;font-size:12px}
.stages{display:flex;flex-direction:column;gap:8px}.st{display:grid;grid-template-columns:22px 1fr auto;gap:8px;align-items:center}
.ic{width:18px;height:18px;border-radius:50%;display:grid;place-items:center;font-size:11px;font-weight:700;border:2px solid var(--muted);color:var(--muted)}
.ic.done{background:var(--good);border-color:var(--good);color:#fff}.ic.running{border-color:var(--s1);color:var(--s1)}.st .x{color:var(--text2);font-size:12px}
label.tg{font-size:12px;color:var(--text2);display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.scroll{overflow-x:auto}
</style></head><body><main>
<header><h1>Item 7 training: live progress</h1><span class="sub" id="sub"></span><span class="live"><span class="dot"></span><span id="clock">connecting…</span></span></header>
<section class="card" style="margin-bottom:12px"><div style="display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline">
 <h2 style="margin:0">Live</h2><span id="phase" class="x" style="color:var(--text2)"></span><span id="saved" style="margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums"></span></div>
 <div style="display:flex;align-items:baseline;gap:10px;margin-top:6px"><span id="liveu" style="font-size:34px;font-weight:650;font-variant-numeric:tabular-nums">–</span><span id="liveu2" style="color:var(--text2)"></span></div>
 <div class="bar" style="height:10px;position:relative"><i id="livebar" style="width:0;transition:width 1s linear"></i></div>
 <div id="evalmarks" style="position:relative;height:14px;font-size:10px;color:var(--muted)"></div>
 <div class="cap" style="margin:10px 0 4px">GPU utilisation, last 6 minutes (sampled every 2 s)</div><div id="gspark"></div></section>
<section class="tiles" id="tiles"></section>
<section class="grid2">
 <div class="card"><h2>Validation balanced accuracy</h2><div class="cap">Every validation check; ● = best checkpoint kept. Dashed = comparison recipe, same seed.</div>
  <div class="legend" id="lg1"></div><div id="c1"></div></div>
 <div class="card"><h2>Validation parse rate</h2><div class="cap">Share of validation answers that parse as valid JSON.</div>
  <div class="legend" id="lg2"></div><div id="c2"></div></div>
</section>
<section class="grid2">
 <div class="card"><h2>Training loss</h2><div class="cap">25-update moving average (tier tokens weighted ×4).</div><div class="legend" id="lg3"></div><div id="c3"></div></div>
 <div class="card"><h2>Queue</h2><div class="cap">Runs one GPU job at a time, in this order.</div><div class="stages" id="stages"></div>
  <div style="margin-top:14px"><label class="tg"><input type="checkbox" id="cmp" checked> Show comparison recipe</label></div></div>
</section>
<section class="card"><h2>Best checkpoint per seed</h2><div class="cap">Balanced accuracy on validation, overall and by window length N.</div><div class="scroll"><table id="tbl"></table></div></section>
</main><div class="tip" id="tip"></div>
<script>
const SEEDS=["101","202","303"],VAR={"101":"--s1","202":"--s2","303":"--s3"};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let D=null,showCmp=true;
document.getElementById("cmp").onchange=e=>{showCmp=e.target.checked;render()};
const tip=document.getElementById("tip");
function fmt(x,d=3){return x==null?"–":(+x).toFixed(d)}
function hm(s){if(s==null)return"–";s=Math.round(s);const h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?`${h} h ${m} min`:`${m} min`}
function legend(id,cmp){const el=document.getElementById(id);el.innerHTML=SEEDS.map(s=>`<span><i class="sw" style="background:var(${VAR[s]})"></i>${D.recipe} seed ${s}</span>`).join("")+(cmp&&showCmp&&D.compare?`<span><i class="sw d" style="border-color:var(--muted)"></i>${D.compare} (same seed)</span>`:"")}
function chart(id,series,{ymin=0,ymax=1,xmax,yfmt=v=>v.toFixed(1),markers=true,best=null,refLine=null}){
 const W=560,H=250,m={l:38,r:64,t:10,b:26},iw=W-m.l-m.r,ih=H-m.t-m.b;
 const X=u=>m.l+u/xmax*iw,Y=v=>m.t+(1-(v-ymin)/(ymax-ymin))*ih;
 let g="";for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4;g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css("--grid")}"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${yfmt(v)}</text>`}
 const step=xmax>1000?300:200;for(let u=0;u<=xmax;u+=step)g+=`<text x="${X(u)}" y="${H-6}" text-anchor="middle">${u}</text>`;
 g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${m.t+ih}" y2="${m.t+ih}" stroke="${css("--border")}"/>`;
 if(refLine!=null)g+=`<line x1="${X(refLine)}" x2="${X(refLine)}" y1="${m.t}" y2="${m.t+ih}" stroke="${css("--muted")}" stroke-dasharray="3 4"/><text x="${X(refLine)+4}" y="${m.t+10}">early stop allowed</text>`;
 const labels=[];
 for(const s of series){if(!s.pts.length)continue;const col=css(s.v);
  const d=s.pts.map((p,i)=>`${i?"L":"M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  g+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="${s.ref?1.5:2}" ${s.ref?'stroke-dasharray="5 4" opacity=".5"':''} stroke-linejoin="round" stroke-linecap="round"/>`;
  if(markers&&!s.ref)for(const p of s.pts){const isB=best&&best[s.seed]===p[0];g+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${isB?5.5:3.5}" fill="${isB?col:css("--surface")}" stroke="${col}" stroke-width="2"/>`}
  if(!s.ref){const p=s.pts[s.pts.length-1];labels.push({y:Y(p[1]),x:X(p[0]),t:`${s.seed} ${yfmt(p[1])}`,c:col})}}
 labels.sort((a,b)=>a.y-b.y);for(let i=1;i<labels.length;i++)if(labels[i].y-labels[i-1].y<13)labels[i].y=labels[i-1].y+13;
 for(const l of labels)g+=`<text x="${W-m.r+6}" y="${l.y+4}" style="fill:${css("--text2")}">${l.t}</text>`;
 g+=`<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" data-hit="1"/>`;
 const el=document.getElementById(id);el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img">${g}<line id="${id}x" y1="${m.t}" y2="${m.t+ih}" stroke="${css("--muted")}" opacity="0"/></svg>`;
 const svg=el.querySelector("svg"),xl=svg.querySelector(`#${id}x`);
 svg.onmousemove=e=>{const r=svg.getBoundingClientRect(),u=((e.clientX-r.left)/r.width*W-m.l)/iw*xmax;
  let rows=[],near=null;for(const s of series){if(!s.pts.length)continue;let bp=null;for(const p of s.pts)if(!bp||Math.abs(p[0]-u)<Math.abs(bp[0]-u))bp=p;
   if(!near||Math.abs(bp[0]-u)<Math.abs(near-u))near=bp[0];rows.push({s,bp})}
  if(near==null)return;rows=rows.filter(r=>Math.abs(r.bp[0]-near)<=(markers?60:15));
  xl.setAttribute("x1",X(near));xl.setAttribute("x2",X(near));xl.setAttribute("opacity",".6");
  tip.innerHTML=`<b>update ${near}</b>`+rows.map(r=>`<div class="r"><span><i class="sw ${r.s.ref?'d':''}" style="${r.s.ref?'border-color':'background'}:${css(r.s.v)}"></i> ${r.s.name}</span><span>${yfmt(r.bp[1])}</span></div>`).join("");
  tip.style.display="block";tip.style.left=Math.min(e.clientX+14,innerWidth-190)+"px";tip.style.top=(e.clientY+14)+"px"};
 svg.onmouseleave=()=>{tip.style.display="none";xl.setAttribute("opacity","0")};
}
function render(){if(!D)return;
 document.getElementById("clock").textContent="updated "+D.time;
 document.getElementById("sub").textContent=`recipe ${D.recipe} · ${D.total_updates} updates max · compared with ${D.compare}`;
 const a=D.active_seed,r=a?D.runs[a]:null,g=D.gpu,e=D.eta;
 const bests=SEEDS.map(s=>D.runs[s]&&D.runs[s].best_val?D.runs[s].best_val.balanced_accuracy:null).filter(x=>x!=null);
 const ls=r&&r.evals.length?r.evals[r.evals.length-1]:null;
 const tiles=[
  {k:"Current run",v:a?`seed ${a}`:(D.stages.find(s=>s.state==="running")||{name:"idle"}).name,n:r?`update ${r.updates} / ${D.total_updates}`:"",p:r?r.updates/D.total_updates:null},
  {k:"Latest check",v:ls?fmt(ls.bal,2):"–",n:ls?`update ${ls.update} · parse ${fmt(ls.parse*100,0)}%`:""},
  {k:"Best this seed",v:r&&r.best_val?fmt(r.best_val.balanced_accuracy,2):"–",n:r?`update ${r.best_update} · ${r.evals_since_best}/3 checks since best`:""},
  {k:`Mean best (${bests.length} seed${bests.length==1?"":"s"})`,v:bests.length?fmt(bests.reduce((x,y)=>x+y,0)/bests.length,2):"–",n:"validation balanced accuracy"},
  {k:"Time left, this seed",v:e?hm(e.to_max):"–",n:e?`to the ${D.total_updates}-update cap; early stop can end it sooner`:"measuring…"},
  {k:"GPU",v:g?`${Math.round(g.util)}%`:"–",n:g?`${(g.mem/1024).toFixed(1)} / ${(g.mem_total/1024).toFixed(0)} GB · ${g.temp}°C · ${Math.round(g.power)} W`:"nvidia-smi unavailable",p:g?g.mem/g.mem_total:null}];
 document.getElementById("tiles").innerHTML=tiles.map(t=>`<div class="card tile"><div class="k">${t.k}</div><div class="v">${t.v}</div><div class="n">${t.n}</div>${t.p!=null?`<div class="bar"><i style="width:${(t.p*100).toFixed(1)}%"></i></div>`:""}</div>`).join("");
 const mk=(key,src,ref)=>SEEDS.map(s=>({seed:s,v:VAR[s],ref,name:`${ref?D.compare:D.recipe} ${s}`,pts:(src[s]?src[s].evals:[]).map(p=>[p.update,p[key]])}));
 const best=Object.fromEntries(SEEDS.map(s=>[s,D.runs[s]?D.runs[s].best_update:null]));
 const cmp=showCmp&&D.compare;
 const xmax=Math.max(D.total_updates,...SEEDS.map(s=>D.reference[s]?D.reference[s].updates:0));
 legend("lg1",1);legend("lg2",1);legend("lg3",0);
 chart("c1",[...(cmp?mk("bal",D.reference,true):[]),...mk("bal",D.runs,false)],{xmax,best,refLine:D.min_updates});
 chart("c2",[...(cmp?mk("parse",D.reference,true):[]),...mk("parse",D.runs,false)],{xmax,yfmt:v=>Math.round(v*100)+"%"});
 const L=SEEDS.map(s=>({seed:s,v:VAR[s],name:`${D.recipe} ${s}`,pts:D.runs[s]?D.runs[s].loss:[]}));
 const lmax=Math.max(1,...L.flatMap(s=>s.pts.map(p=>p[1])));chart("c3",L,{xmax:D.total_updates,ymax:Math.min(lmax,6),markers:false,yfmt:v=>v.toFixed(1)});
 document.getElementById("stages").innerHTML=D.stages.map((s,i)=>{let x=s.state;if(s.progress!=null&&s.state!=="waiting")x+=` · ${Math.round(s.progress*100)}% of max`;
  if(s.detail&&s.detail.arms){x=Object.entries(s.detail.arms).map(([k,v])=>`${k} ${v.done}/${v.total}`).join(" · ")||x}
  return `<div class="st"><span class="ic ${s.state}">${s.state==="done"?"✓":i+1}</span><span>${s.name}</span><span class="x">${x}</span></div>`}).join("");
 const Ns=["1","5","10","20","50"],row=(lab,rr)=>{const b=rr&&rr.best_val;return `<tr><td>${lab}</td><td>${rr?rr.best_update:"–"}</td><td><b>${b?fmt(b.balanced_accuracy,3):"–"}</b></td><td>${b?fmt(b.parse_rate*100,0)+"%":"–"}</td>${Ns.map(n=>`<td>${b&&b.by_n[n]!=null?fmt(b.by_n[n],2):"–"}</td>`).join("")}<td>${rr?(rr.done?"done":"running"):"waiting"}</td></tr>`};
 document.getElementById("tbl").innerHTML=`<tr><th>Run</th><th>Best update</th><th>Bal. acc.</th><th>Parse</th>${Ns.map(n=>`<th>N=${n}</th>`).join("")}<th>Status</th></tr>`+
  SEEDS.map(s=>row(`${D.recipe} seed ${s}`,D.runs[s])).join("")+(D.compare?SEEDS.map(s=>row(`${D.compare} seed ${s}`,D.reference[s])).join(""):"");
}
let skew=0;
async function poll(){try{const t=Date.now()/1000;D=await (await fetch("/api/status",{cache:"no-store"})).json();skew=D.now-t;render();live()}catch(e){document.getElementById("clock").textContent="server not reachable"}}
function live(){if(!D)return;const e=D.eta,now=Date.now()/1000+skew;
 document.getElementById("clock").textContent=new Date((now)*1000).toLocaleTimeString([], {hour12:false});
 const g=D.gpu;if(g&&g.hist.length>1){const W=1100,H=46,h=g.hist,t1=h[h.length-1][0],X=t=>W-(t1-t)/360*W,Y=u=>H-2-u/100*(H-4);
  const d=h.filter(p=>t1-p[0]<=360).map((p,i)=>`${i?"L":"M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  document.getElementById("gspark").innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:46px"><line x1="0" x2="${W}" y1="${Y(100)}" y2="${Y(100)}" stroke="${css("--grid")}"/><line x1="0" x2="${W}" y1="${Y(0)}" y2="${Y(0)}" stroke="${css("--border")}"/><path d="${d}" fill="none" stroke="${css("--s1")}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`}
 if(!e){document.getElementById("liveu").textContent="–";document.getElementById("phase").textContent=(D.stages.find(s=>s.state==="running")||{name:"no training run active"}).name;return}
 const age=now-e.saved_at,nextEval=Math.ceil((e.saved_update+1)/e.eval_every)*e.eval_every;
 const tReach=(nextEval-e.saved_update)*e.sec_per_update,inVal=age>tReach;
 const est=Math.min(D.total_updates,nextEval,Math.floor(e.saved_update+age/e.sec_per_update));
 document.getElementById("liveu").textContent=`≈ ${est}`;
 document.getElementById("liveu2").textContent=`/ ${D.total_updates} updates · seed ${D.active_seed} · last save at update ${e.saved_update}`;
 document.getElementById("livebar").style.width=(est/D.total_updates*100).toFixed(2)+"%";
 document.getElementById("evalmarks").innerHTML=Array.from({length:Math.floor(D.total_updates/e.eval_every)},(_,i)=>(i+1)*e.eval_every).map(u=>`<span style="position:absolute;left:${u/D.total_updates*100}%;transform:translateX(-50%)">│</span>`).join("")+`<span style="position:absolute;left:${D.min_updates/D.total_updates*100}%;transform:translateX(-50%);top:0;color:var(--warn)">▲</span>`;
 document.getElementById("phase").textContent=inVal?`validation check at update ${nextEval} running · ${hm(age-tReach)} elapsed of ≈ ${hm(e.val_sec)}`:`training · next validation check at update ${nextEval} in ≈ ${hm(tReach-age)}`;
 document.getElementById("saved").textContent=`state saved ${Math.round(age)} s ago · ${e.sec_per_update.toFixed(1)} s/update ${e.measured?"measured":"(prior)"} · check ≈ ${hm(e.val_sec)} ${e.val_measured?"measured":"(prior)"}`;
}
poll();setInterval(poll,2000);setInterval(live,1000);matchMedia("(prefers-color-scheme: dark)").addEventListener("change",render);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="r4")
    ap.add_argument("--compare", default="r3")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    Handler.recipe, Handler.compare = a.recipe, a.compare
    print(f"dashboard: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
