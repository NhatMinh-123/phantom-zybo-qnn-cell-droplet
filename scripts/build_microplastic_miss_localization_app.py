"""Build a browser review app that localizes particles in miss-audit frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--review-package",
        type=Path,
        default=ROOT / "review_packages" / "microplastic_qnn_review_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "review_packages" / "microplastic_miss_localization_v1",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, fallback: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float(fallback)


def centered_crop(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT_101)
    x, y = int(round(cx)), int(round(cy))
    crop = padded[y : y + size, x : x + size]
    if crop.shape[:2] != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop.copy()


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Missed Particle Localization</title>
<style>
:root{color-scheme:dark;font-family:Arial,sans-serif;background:#101418;color:#eef2f5}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#101418}
header{height:72px;padding:0 28px;display:flex;align-items:center;gap:18px;border-bottom:1px solid #37414a;background:#161c22}
h1{font-size:22px;margin:0}.progress{font-size:18px;color:#bdd0df}.spacer{flex:1}
button{min-height:46px;padding:0 22px;border:1px solid #52616d;background:#202832;color:#fff;font-size:17px;cursor:pointer}
button:hover{border-color:#86a6bc}button.active{border-color:#f2b51d;color:#ffe192}.primary{border-color:#20a86b}.danger{border-color:#d95b5b}.warn{border-color:#d79a16}
.toolbar{padding:16px 28px;display:flex;gap:10px;border-bottom:1px solid #27313a}
main{padding:24px 28px 96px}.viewer{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;max-width:1240px;margin:auto}
.panel{min-width:0}.panel h2{text-align:center;font-size:16px;font-weight:500;margin:0 0 8px;color:#b9c7d2}
.panel img,.clickable{display:block;width:100%;aspect-ratio:1;border:1px solid #4c5964;background:#090c0f}
.panel img{object-fit:contain}.clickable{position:relative}.clickable img,.clickable canvas{position:absolute;inset:0;width:100%;height:100%}
.clickable img{object-fit:contain}.clickable canvas{cursor:crosshair}.meta{text-align:center;margin:20px 0;color:#c6d3dc;font-size:17px}
.controls{position:fixed;left:0;right:0;bottom:0;min-height:78px;padding:12px 28px;display:flex;align-items:center;justify-content:center;gap:12px;background:#161c22;border-top:1px solid #37414a}
.nav{position:absolute;width:56px;padding:0;font-size:24px}.nav.prev{left:28px}.nav.next{right:28px}
.empty{text-align:center;padding:120px 20px;color:#bbc8d1}.empty[hidden]{display:none!important}
@media(max-width:800px){.viewer{grid-template-columns:1fr}.panel:not(.center){display:none}.controls{padding-left:90px;padding-right:90px;flex-wrap:wrap}button{font-size:14px;padding:0 12px}}
</style></head><body>
<header><h1>Missed Particle Localization</h1><span class="progress" id="progress"></span><span class="spacer"></span><button id="exportCsv">Export CSV</button></header>
<div class="toolbar"><button id="pendingTab" class="active">Pending only</button><button id="undoPoint">Undo point</button><button id="clearPoints">Clear marks</button></div>
<main><section class="viewer" id="viewer">
<div class="panel"><h2>t-1</h2><img id="previousImage" alt="Previous frame"></div>
<div class="panel center"><h2>t0 - click every particle</h2><div class="clickable"><img id="centerImage" alt="Center frame"><canvas id="overlay" width="128" height="128"></canvas></div></div>
<div class="panel"><h2>t+1</h2><img id="nextImage" alt="Next frame"></div>
</section><div class="empty" id="emptyState" hidden>No pending items</div><div class="meta" id="meta"></div></main>
<section class="controls"><button class="nav prev" id="previous" aria-label="Previous">&#8592;</button><button class="primary" id="confirmPoints">Confirm boxes</button><button class="danger" id="noParticle">No particle</button><button class="warn" id="uncertain">Uncertain</button><button class="nav next" id="next" aria-label="Next">&#8594;</button></section>
<script>
const items=__ITEMS__;
const storageKey='microplastic-miss-localization-__PACKAGE_ID__';
let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){saved={}}
let index=0,pendingOnly=true;
const previousImage=document.getElementById('previousImage'),centerImage=document.getElementById('centerImage'),nextImage=document.getElementById('nextImage');
const overlay=document.getElementById('overlay'),ctx=overlay.getContext('2d'),viewer=document.getElementById('viewer'),emptyState=document.getElementById('emptyState'),meta=document.getElementById('meta');
function list(){return pendingOnly?items.filter(item=>!saved[item.review_id]?.label):items}
function state(item){return saved[item.review_id]||{label:'',points:[]}}
function persist(item,value){saved[item.review_id]=value;localStorage.setItem(storageKey,JSON.stringify(saved))}
function draw(){ctx.clearRect(0,0,overlay.width,overlay.height);const current=list();if(!current.length)return;const s=state(current[index]);for(const point of s.points||[]){const x=Number(point.x),y=Number(point.y);ctx.strokeStyle='#16e27a';ctx.lineWidth=1.5;ctx.strokeRect(x-16,y-16,32,32);ctx.beginPath();ctx.moveTo(x-4,y);ctx.lineTo(x+4,y);ctx.moveTo(x,y-4);ctx.lineTo(x,y+4);ctx.stroke()}}
function render(){const current=list();const reviewed=items.filter(item=>saved[item.review_id]?.label).length;document.getElementById('progress').textContent=`${reviewed} / ${items.length} reviewed`;document.getElementById('pendingTab').classList.toggle('active',pendingOnly);if(!current.length){viewer.hidden=true;meta.hidden=true;emptyState.hidden=false;return}viewer.hidden=false;meta.hidden=false;emptyState.hidden=true;index=Math.max(0,Math.min(index,current.length-1));const item=current[index];previousImage.src=item.previous_asset;centerImage.onload=draw;centerImage.src=item.center_asset;nextImage.src=item.next_asset;const s=state(item);meta.textContent=`${item.review_id} | source ${item.source_review_id} | frame ${item.best_frame} | sequence ${item.droplet_sequence} | boxes ${(s.points||[]).length}`;draw()}
overlay.onclick=event=>{const current=list();if(!current.length)return;const item=current[index],rect=overlay.getBoundingClientRect();const x=Math.max(0,Math.min(item.processing_size-1,Math.round((event.clientX-rect.left)*item.processing_size/rect.width)));const y=Math.max(0,Math.min(item.processing_size-1,Math.round((event.clientY-rect.top)*item.processing_size/rect.height)));const s=state(item),points=[...(s.points||[])];const near=points.findIndex(point=>Math.hypot(point.x-x,point.y-y)<=6);if(near>=0)points.splice(near,1);else points.push({x,y});persist(item,{label:'',points});render()};
function finish(label){const current=list();if(!current.length)return;const item=current[index],s=state(item);if(label==='particle'&&!(s.points||[]).length)return;persist(item,{label,points:s.points||[]});if(pendingOnly)index=0;else index=Math.min(index+1,list().length-1);render()}
document.getElementById('confirmPoints').onclick=()=>finish('particle');document.getElementById('noParticle').onclick=()=>finish('no_particle');document.getElementById('uncertain').onclick=()=>finish('uncertain');
document.getElementById('undoPoint').onclick=()=>{const current=list();if(!current.length)return;const item=current[index],s=state(item),points=[...(s.points||[])];points.pop();persist(item,{label:'',points});render()};
document.getElementById('clearPoints').onclick=()=>{const current=list();if(!current.length)return;persist(current[index],{label:'',points:[]});render()};
document.getElementById('pendingTab').onclick=()=>{pendingOnly=!pendingOnly;index=0;render()};document.getElementById('previous').onclick=()=>{index=Math.max(0,index-1);render()};document.getElementById('next').onclick=()=>{index=Math.min(list().length-1,index+1);render()};
document.getElementById('exportCsv').onclick=()=>{const headers=['review_id','source_review_id','best_frame','droplet_sequence','processing_size','reviewed_label','review_status','point_count','points_json'];const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';const lines=[headers.map(quote).join(',')];for(const item of items){const s=state(item);const values={...item,reviewed_label:s.label||'',review_status:s.label?'reviewed':'pending',point_count:(s.points||[]).length,points_json:JSON.stringify(s.points||[])};lines.push(headers.map(header=>quote(values[header])).join(','))}const blob=new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'});const anchor=document.createElement('a');anchor.href=URL.createObjectURL(blob);anchor.download='microplastic_missed_particle_locations.csv';anchor.click();URL.revokeObjectURL(anchor.href)};
render();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    labels = args.labels.expanduser().resolve()
    review_package = args.review_package.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")

    label_rows = read_csv(labels)
    misses = [
        row for row in label_rows
        if row.get("item_type") == "audit"
        and row.get("review_status") == "reviewed"
        and row.get("reviewed_label") == "missed_particle"
    ]
    if not misses:
        raise RuntimeError("No reviewed missed_particle audit rows found")

    package_summary = json.loads((review_package / "review_summary.json").read_text(encoding="utf-8"))
    run_dir = Path(package_summary["source_run"])
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source = Path(package_summary["source_video"])
    configuration = run_summary["configuration"]
    roi = configuration["acquisition_roi"]
    roi_x, roi_y = int(roi["x"]), int(roi["y"])
    roi_w, roi_h = int(roi["width"]), int(roi["height"])
    processing_size = int(configuration["processing_size"])
    frame_rows = {int(row["frame"]): row for row in read_csv(run_dir / "per_frame.csv")}

    frame_count = int(run_summary["source"]["frames_processed"])
    required = {
        max(0, min(frame_count - 1, int(row["best_frame"]) + offset))
        for row in misses for offset in (-1, 0, 1)
    }
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")
    frames: dict[int, np.ndarray] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in required:
            frames[frame_index] = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].copy()
        frame_index += 1
    capture.release()
    missing_frames = required.difference(frames)
    if missing_frames:
        raise RuntimeError(f"Could not decode frames: {sorted(missing_frames)[:10]}")

    assets = output / "items"
    assets.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, row in enumerate(sorted(misses, key=lambda item: int(item["best_frame"])), start=1):
        review_id = f"M{index:04d}"
        center_frame = int(row["best_frame"])
        center_row = frame_rows[center_frame]
        fallback_x = number(center_row, "droplet_center_x", roi_w / 2)
        fallback_y = number(center_row, "droplet_center_y", roi_h / 2)
        paths: dict[int, str] = {}
        for offset in (-1, 0, 1):
            frame = max(0, min(frame_count - 1, center_frame + offset))
            frame_row = frame_rows.get(frame, {})
            cx = number(frame_row, "droplet_center_x", fallback_x)
            cy = number(frame_row, "droplet_center_y", fallback_y)
            crop = centered_crop(frames[frame], cx, cy, processing_size)
            destination = assets / f"{review_id}_{offset:+d}.jpg"
            cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
            paths[offset] = destination.relative_to(output).as_posix()
        manifest.append({
            "review_id": review_id,
            "source_review_id": row["review_id"],
            "best_frame": center_frame,
            "droplet_sequence": int(row["droplet_sequence"]),
            "processing_size": processing_size,
            "previous_asset": paths[-1],
            "center_asset": paths[0],
            "next_asset": paths[1],
        })

    package_id = hashlib.sha256((str(labels) + hashlib.sha256(labels.read_bytes()).hexdigest()).encode()).hexdigest()[:12]
    html = HTML.replace("__ITEMS__", json.dumps(manifest, ensure_ascii=True)).replace("__PACKAGE_ID__", package_id)
    (output / "index.html").write_text(html, encoding="utf-8")
    write_manifest(output / "localization_manifest.csv", manifest)
    summary = {
        "name": "microplastic_miss_localization_v1",
        "source_labels": str(labels),
        "source_labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "source_video": str(source),
        "source_run": str(run_dir),
        "missed_particle_frames": len(manifest),
        "processing_size": processing_size,
        "qnn_patch_size": 32,
        "package_id": package_id,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
