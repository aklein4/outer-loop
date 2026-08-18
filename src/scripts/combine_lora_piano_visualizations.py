"""Combine saved OLoop-LoRA and Piano prediction visualizations.

No model inference is performed. The script reads the manifests and NPZ files
already stored in two result folders and writes one self-contained HTML file::

    python scripts/combine_lora_piano_visualizations.py \
      local_figures/lora_scale_viz/aklein4--horizon-v2_baseline/code_search_net_trajectory \
      local_figures/scale_viz/aklein4--horizon-v2_piano-melted/code_search_net_trajectory

The first positional argument is the OLoop-LoRA result folder and the second is
the Piano result folder. By default the output is ``combined_lora_piano.html``
inside the Piano folder; use ``--output`` to choose another path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lora_folder", type=Path)
    parser.add_argument("piano_folder", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def load_manifest(folder: Path) -> dict[str, Any]:
    path = folder / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text())


def entry_key(entry: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(entry["subset"]),
        str(entry["latent"]),
        int(entry["dataset_index"]),
        int(entry["episode"]),
    )


def index_entries(manifest: dict[str, Any], label: str) -> dict[tuple, dict]:
    indexed: dict[tuple, dict] = {}
    for entry in manifest["entries"]:
        key = entry_key(entry)
        if key in indexed:
            raise ValueError(f"Duplicate {label} entry key: {key}")
        indexed[key] = entry
    return indexed


def resolve_data_file(folder: Path, entry: dict[str, Any]) -> Path:
    relative = Path(entry["data_file"])
    candidates = (folder / relative, folder / "data" / relative)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not resolve {relative}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def quantile(values: np.ndarray, q: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else default


def token_label(piece: Any) -> str:
    return (
        str(piece)
        .lstrip(" ")
        .replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\t", "⇥")
    )


def load_pair(
    lora_folder: Path,
    piano_folder: Path,
    lora_entry: dict[str, Any],
    piano_entry: dict[str, Any],
) -> dict[str, Any]:
    with np.load(resolve_data_file(lora_folder, lora_entry)) as source:
        lora = {name: source[name] for name in source.files}
    with np.load(resolve_data_file(piano_folder, piano_entry)) as source:
        piano = {name: source[name] for name in source.files}

    for field in ("input_ids", "target_ids", "target_assistant"):
        if not np.array_equal(lora[field], piano[field]):
            raise ValueError(
                f"Token alignment differs for {entry_key(piano_entry)} in {field}"
            )
    gates = piano["gates"].astype(np.float64)
    if gates.ndim != 2 or gates.shape[0] == 0:
        raise ValueError(
            f"Piano entry {entry_key(piano_entry)} has no token-gate rows"
        )
    gate_means = np.maximum(gates.mean(axis=1), 1e-30)
    gate_p95 = np.maximum(np.quantile(gates, 0.95, axis=1), 1e-30)

    piano_logp = piano["logp_updated"].astype(np.float64)
    piano_base_logp = piano["logp_empty"].astype(np.float64)
    lora_logp = lora["logp_updated"].astype(np.float64)
    lora_base_logp = lora["logp_empty"].astype(np.float64)
    piano_kl = piano["kl"].astype(np.float64)
    lora_kl = lora["kl"].astype(np.float64)
    piano_delta = piano_logp - piano_base_logp
    lora_delta = lora_logp - lora_base_logp
    cross_delta = piano_logp - lora_logp
    shared_kl = max(
        quantile(np.concatenate((piano_kl, lora_kl)), 0.95, 1.0), 1e-12
    )
    shared_delta = max(
        quantile(
            np.abs(np.concatenate((piano_delta, lora_delta))), 0.95, 1.0
        ),
        1e-12,
    )
    cross_delta_scale = max(
        quantile(np.abs(cross_delta), 0.95, 1.0), 1e-12
    )

    return {
        "subset": piano_entry["subset"],
        "source": piano_entry["source"],
        "latent": piano_entry["latent"],
        "datasetIndex": int(piano_entry["dataset_index"]),
        "episode": int(piano_entry["episode"]),
        "inputTokens": [token_label(piece) for piece in piano["input_pieces"]],
        "targetTokens": [token_label(piece) for piece in piano["target_pieces"]],
        "fullTextPieces": piano["target_pieces"].astype(str).tolist(),
        "assistant": piano["target_assistant"].astype(bool).tolist(),
        "pLogp": piano_logp.tolist(),
        "pBaseLogp": piano_base_logp.tolist(),
        "loraLogp": lora_logp.tolist(),
        "loraBaseLogp": lora_base_logp.tolist(),
        "pKl": piano_kl.tolist(),
        "loraKl": lora_kl.tolist(),
        "pDelta": piano_delta.tolist(),
        "loraDelta": lora_delta.tolist(),
        "crossDelta": cross_delta.tolist(),
        "pRank": piano["rank_updated"].astype(int).tolist(),
        "pBaseRank": piano["rank_empty"].astype(int).tolist(),
        "loraRank": lora["rank_updated"].astype(int).tolist(),
        "loraBaseRank": lora["rank_empty"].astype(int).tolist(),
        "normalizedGates": (gates / gate_means[:, None]).tolist(),
        "gateColors": (gates / gate_p95[:, None]).tolist(),
        "sharedKlScale": shared_kl,
        "sharedDeltaScale": shared_delta,
        "crossDeltaScale": cross_delta_scale,
    }


def pair_examples(lora_folder: Path, piano_folder: Path) -> list[dict[str, Any]]:
    lora_entries = index_entries(load_manifest(lora_folder), "LoRA")
    piano_entries = index_entries(load_manifest(piano_folder), "Piano")
    missing_lora = sorted(set(piano_entries) - set(lora_entries))
    missing_piano = sorted(set(lora_entries) - set(piano_entries))
    if missing_lora or missing_piano:
        raise ValueError(
            "Result folders do not contain identical trajectory episodes. "
            f"Missing from LoRA: {missing_lora[:5]}; "
            f"missing from Piano: {missing_piano[:5]}"
        )
    examples = [
        load_pair(lora_folder, piano_folder, lora_entries[key], piano_entries[key])
        for key in piano_entries
    ]
    examples.sort(
        key=lambda item: (
            item["source"].casefold(),
            item["latent"].casefold(),
            item["datasetIndex"],
            item["episode"],
        )
    )
    if not examples:
        raise ValueError("The result folders contain no examples")
    return examples


def build_html(examples: list[dict[str, Any]], note: str | None) -> str:
    payload = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    note_payload = json.dumps(note or "", ensure_ascii=False).replace("</", "<\\/")
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OLoop and LoRA Prediction Effects</title><style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif}body{margin:0;background:#f6f7fb;color:#172033}header{padding:20px 24px 15px;background:white;border-bottom:1px solid #dfe3eb}h1{margin:0 0 5px;font-size:24px}.controls{display:flex;align-items:center;gap:10px;margin:10px 0 7px}.controls label{color:#465268;font-size:13px;font-weight:600}select{min-width:720px;max-width:94vw;padding:7px 30px 7px 9px;border:1px solid #b9c1ce;border-radius:6px;background:white;color:#172033;font:13px ui-monospace,monospace}.subtitle{color:#667085;font:12px ui-monospace,monospace;margin-bottom:13px}.preamble{max-width:1150px;color:#465268;font-size:14px;line-height:1.48}.preamble p{margin:7px 0}.note{display:none;max-width:1100px;margin:12px 0 0;padding:10px 12px;border-left:4px solid #7863b6;background:#f4f0ff;white-space:pre-wrap;font-size:13px}.roles{display:flex;gap:14px;margin-top:11px;font-size:12px;color:#667085}.role:before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.role.a:before{background:#e28a20}.role.n:before{background:#b13aa3}.fulltext{margin-top:13px;padding:11px 13px;max-height:150px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;border:1px solid #d8dde7;border-radius:7px;font:13px/1.48 ui-monospace,monospace}.fulltext span{cursor:pointer;border-radius:2px;box-decoration-break:clone;-webkit-box-decoration-break:clone}.fulltext .a{--role-color:#e28a20;background:#fff0d8}.fulltext .n{--role-color:#b13aa3;background:#f8e8f5}.fulltext.heatmap-active span{background:var(--token-heat)!important;color:var(--token-ink);box-shadow:inset 0 0 0 2px var(--role-color)}.fulltext .visible-token{outline:1px solid #8b5cf6;outline-offset:-1px}.fulltext .linked-hover{outline:2px solid #d97706;outline-offset:-2px}.fulltext .linked-selected{outline:2px solid #2563eb;outline-offset:-2px}.viewport{overflow:auto;padding:16px 18px 24px;min-height:400px}.matrix{display:flex;width:max-content;align-items:flex-start;background:white;border:1px solid #d8dde7;box-shadow:0 4px 18px #29344d14}.labels{width:154px;flex:none;position:sticky;left:0;z-index:5;background:#f8fafc;border-right:2px solid #aeb7c7}.label-top,.label-bottom{height:76px;box-sizing:border-box;display:flex;align-items:center;justify-content:center;text-align:center;padding:6px;font-size:11px;color:#667085}.row-label{height:24px;display:flex;align-items:center;justify-content:center;border-top:1px solid #e5e8ee;font:10px ui-monospace,monospace;text-align:center;padding:0 4px;cursor:pointer;user-select:none}.row-label:hover,.row-label:focus-visible{background:#ede9fe;outline:2px solid #8b5cf6;outline-offset:-2px}.row-label.context-selected{background:#ddd6fe;color:#4c1d95;font-weight:700;box-shadow:inset 3px 0 #7c3aed}.row-label.diag{height:28px;font-weight:600}.label-bottom{border-top:2px solid #aeb7c7}.token-col{width:72px;flex:none;border-right:1px solid #e6e9ef}.token-box{height:76px;box-sizing:border-box;padding:5px 3px;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden;background:#fbfcfe;cursor:pointer}.target{border-top:4px solid #b13aa3}.target.a{border-top-color:#e28a20;background:#fffbf5}.input{border-top:2px solid #aeb7c7;background:#f8fafc}.token-box:hover,.token-box.linked-hover{background:#fef3c7;box-shadow:inset 0 0 0 2px #d97706}.token-box.linked-selected{background:#dbeafe;box-shadow:inset 0 0 0 2px #2563eb}.token{white-space:normal;overflow:hidden;overflow-wrap:anywhere;max-height:50px;line-height:1.15;text-align:center;font:12px ui-monospace,monospace}.index{font:9px ui-monospace,monospace;color:#7b8494;margin-top:3px}.cell{height:24px;box-sizing:border-box;border-top:1px solid #ffffff73;cursor:crosshair}.cell.diag{height:28px}.cell:hover{outline:2px solid #111827;outline-offset:-2px}.tip{position:fixed;pointer-events:none;display:none;z-index:20;background:#111827;color:white;padding:7px 9px;border-radius:6px;white-space:pre;font:12px/1.45 ui-monospace,monospace;box-shadow:0 4px 14px #0005}
</style></head><body><header><h1>OLoop and LoRA Prediction Effects</h1>
<div class="controls"><label for="example">Example</label><select id="example"></select></div><div class="subtitle" id="subtitle"></div>
<div class="preamble"><p>The first eight rows compare next-token predictions under OLoop and LoRA fast weights and their corresponding empty fast-weight baselines. The remaining rows show OLoop's per-token adaptive learning-rate gates.</p>
<p>All probability rows use fixed linear bounds from 0 (white) to 1 (dark green). Both KL rows share one 95th-percentile scale within an example. The two within-model delta-log-probability rows share one symmetric 95th-percentile scale; Δ log p vs lora uses its own symmetric 95th-percentile scale. Gate rows use each layer's 95th percentile.</p>
<p>Click a row label to apply its gradient to the context. Hover a token or diagnostic cell for both models' probabilities, log probabilities, KL values, deltas, and ranks. Click a token in the context or table to jump to that token in the other area.</p></div><div class="note" id="note"></div>
<div class="roles"><span class="role a">assistant target</span><span class="role n">non-assistant target</span></div><div class="fulltext" id="fulltext"></div></header>
<div class="viewport" id="viewport"><div class="matrix" id="matrix"></div></div><div class="tip" id="tip"></div><script>
const EXAMPLES=__PAYLOAD__,NOTE=__NOTE__,DIAG_ROWS=9;
const picker=document.getElementById('example'),matrix=document.getElementById('matrix'),tip=document.getElementById('tip'),fulltext=document.getElementById('fulltext'),subtitle=document.getElementById('subtitle'),viewport=document.getElementById('viewport'),note=document.getElementById('note');if(NOTE){note.textContent=NOTE;note.style.display='block'}
const nameWidth=Math.max(...EXAMPLES.map(d=>d.latent.length)),trajectoryWidth=Math.max(...EXAMPLES.map(d=>String(d.datasetIndex).length)),episodeWidth=Math.max(...EXAMPLES.map(d=>String(d.episode).length)),fixed=s=>s.replaceAll(' ','\u00a0'),groups=new Map();EXAMPLES.forEach((d,i)=>{if(!groups.has(d.source)){const g=document.createElement('optgroup');g.label=d.source;groups.set(d.source,g);picker.appendChild(g)}const o=document.createElement('option');o.value=i;o.textContent=fixed(`${d.latent.padEnd(nameWidth)}  ·  trajectory ${String(d.datasetIndex).padStart(trajectoryWidth)}  ·  episode ${String(d.episode).padStart(episodeWidth)}`);groups.get(d.source).appendChild(o)});
function blend(lo,hi,t){t=Math.max(0,Math.min(1,t));return `rgb(${lo.map((x,i)=>Math.round(x+(hi[i]-x)*t)).join(',')})`}const positive=t=>blend([255,255,255],[20,105,75],t),gate=v=>blend([255,255,255],[160,24,37],v);function diverging(t){t=Math.max(0,Math.min(1,t));return t<.5?blend([49,130,189],[255,255,255],t*2):blend([255,255,255],[20,105,75],(t-.5)*2)}const delta=(v,s)=>diverging(.5+.5*Math.max(-1,Math.min(1,v/s)));function ink(color){const rgb=(color.match(/\d+/g)||[255,255,255]).map(Number),l=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];return l<145?'#fff':'#172033'}
function rowColor(D,row,j){if(row===0)return positive(Math.exp(D.pLogp[j]));if(row===1)return positive(Math.exp(D.pBaseLogp[j]));if(row===2)return positive(Math.exp(D.loraLogp[j]));if(row===3)return positive(Math.exp(D.loraBaseLogp[j]));if(row===4)return positive(D.pKl[j]/D.sharedKlScale);if(row===5)return positive(D.loraKl[j]/D.sharedKlScale);if(row===6)return delta(D.pDelta[j],D.sharedDeltaScale);if(row===7)return delta(D.loraDelta[j],D.sharedDeltaScale);if(row===8)return delta(D.crossDelta[j],D.crossDeltaScale);const layer=D.normalizedGates.length-1-(row-DIAG_ROWS);return gate(D.gateColors[layer][j])}
function diagnosticTip(D,j){return `Δ log p vs LoRA: ${D.crossDelta[j].toFixed(4)}\n\nOLoop\n  p:             ${Math.exp(D.pLogp[j]).toPrecision(6)}\n  p base:        ${Math.exp(D.pBaseLogp[j]).toPrecision(6)}\n  log p:         ${D.pLogp[j].toFixed(4)}\n  log p base:    ${D.pBaseLogp[j].toFixed(4)}\n  KL:            ${D.pKl[j].toFixed(6)}\n  Δ log p:       ${D.pDelta[j].toFixed(4)}\n  rank:          ${D.pRank[j].toLocaleString()}\n  rank base:     ${D.pBaseRank[j].toLocaleString()}\n\nLoRA\n  p:             ${Math.exp(D.loraLogp[j]).toPrecision(6)}\n  p base:        ${Math.exp(D.loraBaseLogp[j]).toPrecision(6)}\n  log p:         ${D.loraLogp[j].toFixed(4)}\n  log p base:    ${D.loraBaseLogp[j].toFixed(4)}\n  KL:            ${D.loraKl[j].toFixed(6)}\n  Δ log p:       ${D.loraDelta[j].toFixed(4)}\n  rank:          ${D.loraRank[j].toLocaleString()}\n  rank base:     ${D.loraBaseRank[j].toLocaleString()}`}
function attachTip(el,text){el.onmouseenter=()=>{tip.style.display='block';tip.textContent=text};el.onmousemove=e=>{tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'};el.onmouseleave=()=>tip.style.display='none'}
let selectedToken=-1,selectedRow=-1,visibleFrame=0;const contextTokens=()=>Array.from(fulltext.querySelectorAll('[data-token-index]')),columns=()=>Array.from(matrix.querySelectorAll('.token-col')),target=i=>columns()[i]?.querySelector('.target');function mirror(i,on){contextTokens()[i]?.classList.toggle('linked-hover',on);target(i)?.classList.toggle('linked-hover',on)}function scrollTable(i){const c=columns()[i];if(c)viewport.scrollTo({left:Math.max(0,c.offsetLeft-viewport.clientWidth/2+c.offsetWidth/2),behavior:'smooth'})}function scrollContext(i){const s=contextTokens()[i];if(s)fulltext.scrollTo({top:Math.max(0,s.offsetTop-fulltext.clientHeight/2+s.offsetHeight/2),behavior:'smooth'})}function selectToken(i,origin){const spans=contextTokens();if(selectedToken>=0){spans[selectedToken]?.classList.remove('linked-selected');target(selectedToken)?.classList.remove('linked-selected')}selectedToken=i;spans[i]?.classList.add('linked-selected');target(i)?.classList.add('linked-selected');origin==='context'?scrollTable(i):scrollContext(i)}function link(el,i,origin){el.addEventListener('mouseenter',()=>mirror(i,true));el.addEventListener('mouseleave',()=>mirror(i,false));el.addEventListener('click',()=>selectToken(i,origin))}
function selectContextRow(D,labels,row){const ls=Array.from(labels.querySelectorAll('.row-label'));if(selectedRow===row){selectedRow=-1;fulltext.classList.remove('heatmap-active');ls[row]?.classList.remove('context-selected');contextTokens().forEach(s=>{s.style.removeProperty('--token-heat');s.style.removeProperty('--token-ink')});return}if(selectedRow>=0)ls[selectedRow]?.classList.remove('context-selected');selectedRow=row;ls[row]?.classList.add('context-selected');fulltext.classList.add('heatmap-active');contextTokens().forEach((s,j)=>{const c=rowColor(D,row,j);s.style.setProperty('--token-heat',c);s.style.setProperty('--token-ink',ink(c))})}function activateLabels(D,labels){Array.from(labels.querySelectorAll('.row-label')).forEach((label,row)=>{label.tabIndex=0;label.setAttribute('role','button');label.title='Apply this row color scale to the context';label.onclick=()=>selectContextRow(D,labels,row);label.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectContextRow(D,labels,row)}}})}
function updateVisible(){visibleFrame=0;const spans=contextTokens(),cols=columns(),labels=matrix.querySelector('.labels'),r=viewport.getBoundingClientRect(),left=Math.max(r.left,labels?labels.getBoundingClientRect().right:r.left);spans.forEach(s=>s.classList.remove('visible-token'));cols.forEach((c,i)=>{const x=c.getBoundingClientRect();if(x.right>left&&x.left<r.right)spans[i]?.classList.add('visible-token')})}function scheduleVisible(){if(!visibleFrame)visibleFrame=requestAnimationFrame(updateVisible)}
function render(index){const D=EXAMPLES[index];selectedToken=-1;selectedRow=-1;tip.style.display='none';matrix.replaceChildren();fulltext.replaceChildren();fulltext.classList.remove('heatmap-active');fulltext.scrollTop=0;viewport.scrollLeft=0;subtitle.textContent=`Episode ${D.episode} · Source: ${D.source} · Repository: ${D.latent} · Trajectory ${D.datasetIndex}`;D.fullTextPieces.forEach((piece,j)=>{const s=document.createElement('span');s.className=D.assistant[j]?'a':'n';s.dataset.tokenIndex=j;s.textContent=piece;link(s,j,'context');attachTip(s,diagnosticTip(D,j));fulltext.appendChild(s)});const labels=document.createElement('div');labels.className='labels';labels.innerHTML='<div class="label-top">token predicted<br>at top</div>'+['p','p_base','p_lora','p_lora_base','KL(p ‖ p_base)','KL(p_lora ‖ p_lora_base)','Δ log p vs base','Δ log p_lora vs lora_base','Δ log p vs lora'].map(x=>`<div class="row-label diag">${x}</div>`).join('')+Array.from({length:D.normalizedGates.length},(_,i)=>`<div class="row-label">layer ${D.normalizedGates.length-1-i}</div>`).join('')+'<div class="label-bottom">token taken in<br>at bottom</div>';matrix.appendChild(labels);activateLabels(D,labels);D.targetTokens.forEach((text,j)=>{const col=document.createElement('div');col.className='token-col';const top=document.createElement('div');top.className='token-box target'+(D.assistant[j]?' a':'');top.innerHTML='<div class="token"></div><div class="index">'+(j+1)+'</div>';top.querySelector('.token').textContent=text;link(top,j,'table');col.appendChild(top);for(let row=0;row<DIAG_ROWS;row++){const cell=document.createElement('div');cell.className='cell diag';cell.style.background=rowColor(D,row,j);attachTip(cell,diagnosticTip(D,j));col.appendChild(cell)}for(let layer=D.normalizedGates.length-1;layer>=0;layer--){const cell=document.createElement('div');cell.className='cell';cell.style.background=gate(D.gateColors[layer][j]);attachTip(cell,D.normalizedGates[layer][j].toFixed(3));col.appendChild(cell)}const bottom=document.createElement('div');bottom.className='token-box input';bottom.innerHTML='<div class="token"></div><div class="index">'+j+'</div>';bottom.querySelector('.token').textContent=D.inputTokens[j];if(j>0)link(bottom,j-1,'table');else bottom.style.cursor='default';col.appendChild(bottom);matrix.appendChild(col)});scheduleVisible()}
viewport.addEventListener('scroll',scheduleVisible,{passive:true});window.addEventListener('resize',scheduleVisible,{passive:true});picker.addEventListener('change',()=>render(Number(picker.value)));render(0);
</script></body></html>'''.replace("__PAYLOAD__", payload).replace("__NOTE__", note_payload)


def main() -> None:
    args = parse_args()
    lora_folder = args.lora_folder.resolve()
    piano_folder = args.piano_folder.resolve()
    examples = pair_examples(lora_folder, piano_folder)
    output = (
        args.output.resolve()
        if args.output is not None
        else piano_folder / "combined_lora_piano.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(examples, args.note))
    print(f"Paired {len(examples)} episodes")
    print(output)


if __name__ == "__main__":
    main()
