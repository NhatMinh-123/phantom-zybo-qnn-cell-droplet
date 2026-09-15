"""Build an offline browser review app for microplastic QNN candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--audit-count", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else default


def centered_crop(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(
        image, half, half, half, half, cv2.BORDER_REFLECT_101
    )
    x = int(round(cx))
    y = int(round(cy))
    crop = padded[y : y + size, x : x + size]
    if crop.shape[:2] != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop.copy()


def enhance_patch(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (2, 98))
    if high - low < 2:
        return gray.copy()
    scaled = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
    return scaled.astype(np.uint8)


def candidate_priority(gray: np.ndarray, row: dict[str, str]) -> tuple[float, str]:
    """Rank faint dark particles ahead of low-contrast detector responses."""
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius2 = (xx - cx) ** 2 + (yy - cy) ** 2
    center = radius2 <= 5.5 ** 2
    ring = (radius2 >= 7.0 ** 2) & (radius2 <= 13.0 ** 2)
    image = gray.astype(np.float32)
    dark_contrast = float(np.median(image[ring]) - np.mean(image[center]))
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    blackhat_peak = float(np.percentile(blackhat[center], 95))
    gradient = float(cv2.Laplacian(image, cv2.CV_32F)[center].std())
    hits = number(row, "hits")
    score = (
        2.2 * max(dark_contrast, 0.0)
        + 1.3 * blackhat_peak
        + 0.35 * gradient
        + 3.0 * min(hits, 5.0)
        + 4.0 * number(row, "max_score")
    )
    tier = "high" if score >= 55.0 else "medium" if score >= 30.0 else "low"
    return round(score, 3), tier


def panel(image: np.ndarray, label: str, size: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    result = np.full((size + 26, size, 3), 24, dtype=np.uint8)
    result[26:] = resized
    cv2.putText(
        result, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (235, 235, 235), 1, cv2.LINE_AA,
    )
    return result


def add_footer(image: np.ndarray, text: str) -> np.ndarray:
    footer = np.full((30, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        footer, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (220, 220, 220), 1, cv2.LINE_AA,
    )
    return np.concatenate([image, footer], axis=0)


def processing_crop(
    acquisition_frames: dict[int, np.ndarray],
    frame_rows: dict[int, dict[str, str]],
    frame_index: int,
    fallback: tuple[float, float],
    size: int,
) -> np.ndarray:
    frame = acquisition_frames[frame_index]
    row = frame_rows.get(frame_index, {})
    cx = number(row, "droplet_center_x", fallback[0])
    cy = number(row, "droplet_center_y", fallback[1])
    return centered_crop(frame, cx, cy, size)


def mark_candidate(image: np.ndarray, x: float, y: float) -> np.ndarray:
    marked = image.copy()
    cx, cy = int(round(x)), int(round(y))
    cv2.rectangle(marked, (cx - 7, cy - 7), (cx + 7, cy + 7), (255, 0, 255), 2)
    cv2.drawMarker(marked, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 9, 1)
    return marked


def select_audit_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    eligible = [
        row for row in rows
        if int(number(row, "droplet_complete")) == 1
        and int(number(row, "candidate_count")) == 0
    ]
    groups: dict[int, list[dict[str, str]]] = {}
    for row in eligible:
        groups.setdefault(int(number(row, "droplet_sequence")), []).append(row)
    selected: list[dict[str, str]] = []
    quota = max(1, math.ceil(count / max(len(groups), 1)))
    for sequence in sorted(groups):
        group = sorted(groups[sequence], key=lambda item: int(item["frame"]))
        indexes = np.linspace(0, len(group) - 1, min(quota, len(group))).astype(int)
        selected.extend(group[index] for index in indexes)
    unique = {int(row["frame"]): row for row in selected}
    if len(unique) < count:
        for row in eligible:
            unique.setdefault(int(row["frame"]), row)
            if len(unique) >= count:
                break
    return sorted(unique.values(), key=lambda row: int(row["frame"]))[:count]


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microplastic QNN Review</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#e9edf2;background:#111418;letter-spacing:0}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#111418}
header{height:64px;display:flex;align-items:center;gap:18px;padding:0 22px;border-bottom:1px solid #303741;background:#171b20}
h1{font-size:18px;margin:0;white-space:nowrap}.progress{color:#aeb8c5;font-variant-numeric:tabular-nums}
.spacer{flex:1}button{font:inherit;color:inherit;background:#242a31;border:1px solid #3b4550;border-radius:6px;min-height:38px;padding:8px 14px;cursor:pointer}
button:hover{background:#303842}button.active{border-color:#f0c64a;color:#fff}.export{background:#185d44;border-color:#258360}
.tabs{display:flex;gap:6px;padding:14px 22px 0}.tabs button{min-width:130px}
main{display:grid;grid-template-rows:1fr auto;min-height:calc(100vh - 118px);padding:18px 22px 22px;gap:16px}
.viewer{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:0}
.viewer img{display:block;max-width:100%;max-height:calc(100vh - 290px);object-fit:contain;border:1px solid #3a424c;background:#090b0d}
.meta{margin-top:10px;color:#b7c0cb;font-size:14px;font-variant-numeric:tabular-nums}
.controls{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:14px}
.labels{display:flex;justify-content:center;gap:10px;flex-wrap:wrap}.labels button{min-width:150px;font-weight:600}
.particle{border-color:#2aa876}.background{border-color:#4b91d1}.uncertain{border-color:#d69b32}.missed{border-color:#df6159}.clear{border-color:#2aa876}
.nav{width:44px;padding:6px;font-size:22px}.empty{text-align:center;color:#aeb8c5;padding:60px}[hidden]{display:none!important}
@media(max-width:760px){header{padding:0 12px;gap:9px}h1{font-size:15px}.progress{display:none}.tabs,main{padding-left:12px;padding-right:12px}.controls{grid-template-columns:44px 1fr 44px}.labels button{min-width:110px}.export{padding:7px 9px}}
</style>
</head>
<body>
<header><h1>Microplastic QNN Review</h1><span class="progress" id="progress"></span><span class="spacer"></span><button id="clearLabel">Clear label</button><button class="export" id="exportCsv">Export CSV</button></header>
<div class="tabs"><button id="candidateTab">Candidate</button><button id="auditTab">Miss audit</button><button id="pendingTab">Pending only</button></div>
<main><section class="viewer" id="viewer"><img id="image" alt="Review item"><div class="meta" id="meta"></div><div class="empty" id="emptyState" hidden>No items in this view</div></section><section class="controls"><button class="nav" id="previous" aria-label="Previous">&#8592;</button><div class="labels" id="labels"></div><button class="nav" id="next" aria-label="Next">&#8594;</button></section></main>
<script>
const items=__ITEMS__;
const storageKey='microplastic-qnn-review-__PACKAGE_ID__';
let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){saved={}}
let mode='candidate',pendingOnly=true,index=0;
const image=document.getElementById('image'),meta=document.getElementById('meta'),labels=document.getElementById('labels'),emptyState=document.getElementById('emptyState');
function list(){let result=items.filter(x=>x.item_type===mode);if(pendingOnly)result=result.filter(x=>!saved[x.review_id]);if(mode==='candidate')result.sort((a,b)=>(b.priority_score||0)-(a.priority_score||0));return result}
function options(){return mode==='candidate'?[['particle','Particle','particle'],['background','Background','background'],['uncertain','Uncertain','uncertain']]:[['clear','Clear','clear'],['missed_particle','Missed particle','missed'],['uncertain','Uncertain','uncertain']]}
function counts(){const relevant=items.filter(x=>x.item_type===mode),done=relevant.filter(x=>saved[x.review_id]).length;return [done,relevant.length]}
function render(){const current=list();const count=counts();document.getElementById('progress').textContent=`${count[0]} / ${count[1]} reviewed`;document.getElementById('candidateTab').classList.toggle('active',mode==='candidate');document.getElementById('auditTab').classList.toggle('active',mode==='audit');document.getElementById('pendingTab').classList.toggle('active',pendingOnly);if(!current.length){image.style.display='none';meta.style.display='none';emptyState.hidden=false;labels.innerHTML='';return}emptyState.hidden=true;image.style.display='block';meta.style.display='block';index=Math.max(0,Math.min(index,current.length-1));const item=current[index];image.src=item.asset;image.style.display='block';meta.textContent=mode==='candidate'?`${item.review_id} | frame ${item.best_frame} | sequence ${item.droplet_sequence} | hits ${item.hits} | priority ${item.priority_tier} ${Number(item.priority_score).toFixed(1)}`:`${item.review_id} | frame ${item.best_frame} | sequence ${item.droplet_sequence}`;labels.innerHTML='';for(const [value,title,klass] of options()){const b=document.createElement('button');b.textContent=title;b.className=klass+(saved[item.review_id]===value?' active':'');b.onclick=()=>setLabel(value);labels.appendChild(b)}}
function setLabel(value){const current=list();if(!current.length)return;const id=current[index].review_id;saved[id]=value;localStorage.setItem(storageKey,JSON.stringify(saved));if(pendingOnly)index=0;else index=Math.min(index+1,list().length-1);render()}
function move(delta){const current=list();if(!current.length)return;index=(index+delta+current.length)%current.length;render()}
document.getElementById('previous').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
document.getElementById('candidateTab').onclick=()=>{mode='candidate';index=0;render()};document.getElementById('auditTab').onclick=()=>{mode='audit';index=0;render()};document.getElementById('pendingTab').onclick=()=>{pendingOnly=!pendingOnly;index=0;render()};
document.getElementById('clearLabel').onclick=()=>{const current=list();if(current.length){delete saved[current[index].review_id];localStorage.setItem(storageKey,JSON.stringify(saved));render()}};
document.getElementById('exportCsv').onclick=()=>{const headers=Object.keys(items[0]).concat(['reviewed_label','review_status']);const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const lines=[headers.map(quote).join(',')];for(const item of items){lines.push(headers.map(h=>quote(h==='reviewed_label'?(saved[item.review_id]||''):h==='review_status'?(saved[item.review_id]?'reviewed':'pending'):item[h])).join(','))}const blob=new Blob([lines.join('\r\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='microplastic_qnn_review_labels.csv';a.click();URL.revokeObjectURL(a.href)};
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key==='1')setLabel(options()[0][0]);if(e.key==='2')setLabel(options()[1][0]);if(e.key==='3')setLabel(options()[2][0])});
render();
</script>
</body></html>'''


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        existing = list(output.iterdir())
        resumable = (
            len(existing) == 1
            and existing[0].name == "items"
            and existing[0].is_dir()
            and not any(existing[0].iterdir())
        )
        if existing and not resumable:
            raise FileExistsError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "items"
    assets.mkdir(exist_ok=True)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    source = (args.source or Path(summary["source"]["path"])).expanduser().resolve()
    config = summary["configuration"]
    roi = config["acquisition_roi"]
    roi_x, roi_y = int(roi["x"]), int(roi["y"])
    roi_w, roi_h = int(roi["width"]), int(roi["height"])
    size = int(config["processing_size"])

    tracks = [
        row for row in read_rows(run_dir / "particle_tracks.csv")
        if row.get("review_patch") and Path(row["review_patch"]).is_file()
    ]
    tracks.sort(key=lambda row: (int(row["best_frame"]), int(row["particle_track"])))
    if args.max_candidates > 0:
        tracks = tracks[: args.max_candidates]
    frame_list = read_rows(run_dir / "per_frame.csv")
    frame_rows = {int(row["frame"]): row for row in frame_list}
    audits = select_audit_rows(frame_list, args.audit_count)

    required: set[int] = set()
    total_frames = int(summary["source"]["frames_processed"])
    review_frames = [int(row["best_frame"]) for row in tracks]
    review_frames.extend(int(row["frame"]) for row in audits)
    for frame in review_frames:
        required.update(
            max(0, min(total_frames - 1, frame + offset))
            for offset in (-1, 0, 1)
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source}")
    acquisition_frames: dict[int, np.ndarray] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in required:
            acquisition_frames[frame_index] = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].copy()
        frame_index += 1
    capture.release()
    missing = required.difference(acquisition_frames)
    if missing:
        raise RuntimeError(f"Could not decode required frames: {sorted(missing)[:10]}")

    manifest: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(tracks))
    for review_index, track_index in enumerate(order, start=1):
        row = tracks[int(track_index)]
        review_id = f"C{review_index:04d}"
        best_frame = int(row["best_frame"])
        best_x, best_y = number(row, "best_x"), number(row, "best_y")
        center_row = frame_rows[best_frame]
        fallback = (
            number(center_row, "droplet_center_x", roi_w / 2),
            number(center_row, "droplet_center_y", roi_h / 2),
        )
        context = []
        for offset in (-1, 0, 1):
            frame = max(0, min(total_frames - 1, best_frame + offset))
            crop = processing_crop(acquisition_frames, frame_rows, frame, fallback, size)
            if offset == 0:
                crop = mark_candidate(crop, best_x, best_y)
            context.append(panel(crop, f"t{offset:+d} frame {frame}", 160))
        patch_path = Path(row["review_patch"])
        patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise RuntimeError(f"Could not read patch: {patch_path}")
        priority_score, priority_tier = candidate_priority(patch, row)
        composite = np.concatenate(
            [panel(patch, "raw 32x32", 160), panel(enhance_patch(patch), "contrast", 160), *context], axis=1
        )
        composite = add_footer(
            composite,
            f"{review_id}  seq={row['droplet_sequence']} track={row['particle_track']} hits={row['hits']}",
        )
        asset = assets / f"{review_id}.jpg"
        cv2.imwrite(str(asset), composite, [cv2.IMWRITE_JPEG_QUALITY, 94])
        manifest.append({
            "review_id": review_id, "item_type": "candidate",
            "droplet_sequence": row["droplet_sequence"], "track_id": row["particle_track"],
            "first_frame": row["first_frame"], "last_frame": row["last_frame"],
            "best_frame": best_frame, "best_x": best_x, "best_y": best_y,
            "hits": row["hits"], "mean_score": row["mean_score"],
            "max_score": row["max_score"], "temporal_confidence": row["confidence"],
            "priority_score": priority_score, "priority_tier": priority_tier,
            "asset": asset.relative_to(output).as_posix(), "source_patch": str(patch_path.resolve()),
        })

    for audit_index, row in enumerate(audits, start=1):
        review_id = f"A{audit_index:04d}"
        best_frame = int(row["frame"])
        fallback = (number(row, "droplet_center_x", roi_w / 2), number(row, "droplet_center_y", roi_h / 2))
        panels = []
        for offset in (-1, 0, 1):
            frame = max(0, min(total_frames - 1, best_frame + offset))
            crop = processing_crop(acquisition_frames, frame_rows, frame, fallback, size)
            panels.append(panel(crop, f"t{offset:+d} frame {frame}", 220))
        composite = add_footer(np.concatenate(panels, axis=1), f"{review_id}  miss audit  seq={row['droplet_sequence']}")
        asset = assets / f"{review_id}.jpg"
        cv2.imwrite(str(asset), composite, [cv2.IMWRITE_JPEG_QUALITY, 94])
        manifest.append({
            "review_id": review_id, "item_type": "audit",
            "droplet_sequence": row["droplet_sequence"], "track_id": "",
            "first_frame": best_frame, "last_frame": best_frame, "best_frame": best_frame,
            "best_x": "", "best_y": "", "hits": "", "mean_score": "",
            "max_score": "", "temporal_confidence": "",
            "asset": asset.relative_to(output).as_posix(), "source_patch": "",
        })

    package_id = hashlib.sha256((str(source) + str(run_dir)).encode()).hexdigest()[:12]
    html = HTML.replace("__ITEMS__", json.dumps(manifest, ensure_ascii=True)).replace("__PACKAGE_ID__", package_id)
    (output / "index.html").write_text(html, encoding="utf-8")
    write_manifest(output / "review_manifest.csv", manifest)
    package_summary = {
        "source_video": str(source), "source_run": str(run_dir),
        "candidate_items": len(tracks), "miss_audit_items": len(audits),
        "total_items": len(manifest), "package_id": package_id,
        "labels": {
            "candidate": ["particle", "background", "uncertain"],
            "audit": ["clear", "missed_particle", "uncertain"],
        },
    }
    (output / "review_summary.json").write_text(json.dumps(package_summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Microplastic QNN review\n\nOpen `index.html`, review both queues, then use `Export CSV`.\n"
        "Candidate labels: `particle`, `background`, `uncertain`.\n"
        "Miss-audit labels: `clear`, `missed_particle`, `uncertain`.\n"
        "Do not use `uncertain` samples for training.\n",
        encoding="utf-8",
    )
    print(json.dumps(package_summary, indent=2))
    print(f"Review app: {output / 'index.html'}")


if __name__ == "__main__":
    main()