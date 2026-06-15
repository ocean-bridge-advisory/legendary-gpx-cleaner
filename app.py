import os
import uuid
import json
import math
import threading
import io
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import gpxpy
import gpxpy.gpx

app = Flask(__name__)
CORS(app)

# In-memory job store
jobs = {}

# ---------------------------------------------------------------------------
# Haversine distance (metres)
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def total_distance_m(points):
    """points: list of (lat, lon)"""
    d = 0.0
    for i in range(1, len(points)):
        d += haversine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
    return d


# ---------------------------------------------------------------------------
# GPX parse / build
# ---------------------------------------------------------------------------

def parse_gpx(gpx_bytes):
    """Return (gpx_obj, trackpoints_list).
    trackpoints_list: list of dicts with lat, lon, ele, time, extensions_xml
    """
    gpx = gpxpy.parse(io.BytesIO(gpx_bytes))
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                ext_xml = None
                if pt.extensions:
                    import xml.etree.ElementTree as ET
                    ext_xml = "".join(ET.tostring(e, encoding="unicode") for e in pt.extensions)
                points.append({
                    "lat": pt.latitude,
                    "lon": pt.longitude,
                    "ele": pt.elevation,
                    "time": pt.time,
                    "ext_xml": ext_xml,
                })
    return gpx, points


def build_gpx(original_gpx, result_points):
    """Build a new GPX preserving metadata, using result_points for coordinates."""
    new_gpx = gpxpy.gpx.GPX()
    new_gpx.name = original_gpx.name
    new_gpx.description = original_gpx.description
    new_gpx.author_name = original_gpx.author_name
    new_gpx.author_email = original_gpx.author_email
    new_gpx.link = original_gpx.link
    new_gpx.link_text = original_gpx.link_text
    new_gpx.time = original_gpx.time
    new_gpx.keywords = original_gpx.keywords

    track = gpxpy.gpx.GPXTrack()
    if original_gpx.tracks:
        track.name = original_gpx.tracks[0].name
        track.type = original_gpx.tracks[0].type
    new_gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    import xml.etree.ElementTree as ET

    for rp in result_points:
        pt = gpxpy.gpx.GPXTrackPoint(
            latitude=rp["lat"],
            longitude=rp["lon"],
            elevation=rp["ele"],
            time=rp["time"],
        )
        if rp.get("ext_xml"):
            try:
                # Wrap in a dummy root to allow multiple children
                wrapped = ET.fromstring(f"<root>{rp['ext_xml']}</root>")
                for child in wrapped:
                    pt.extensions.append(child)
            except Exception:
                pass
        segment.points.append(pt)

    return new_gpx.to_xml()


# ---------------------------------------------------------------------------
# Stateless GPX building (no remembered job state)
# ---------------------------------------------------------------------------

def _iso(dt):
    return dt.isoformat() if dt is not None else None


def _parse_iso(s):
    if not s:
        return None
    try:
        from datetime import datetime as _dt
        return _dt.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def extract_meta(gpx):
    """Pull GPX metadata into a JSON-safe dict the frontend can hold and return."""
    trk = gpx.tracks[0] if gpx.tracks else None
    return {
        "name":         gpx.name,
        "description":  gpx.description,
        "author_name":  gpx.author_name,
        "author_email": gpx.author_email,
        "link":         gpx.link,
        "link_text":    gpx.link_text,
        "time":         _iso(gpx.time),
        "keywords":     gpx.keywords,
        "track_name":   trk.name if trk else None,
        "track_type":   trk.type if trk else None,
    }


def build_gpx_stateless(track, meta):
    """Build GPX from plain dicts + a meta dict. Needs no original GPX object.

    track: [{lat, lon, ele, time(ISO str|None), ext_xml(str|None)}]
    """
    import xml.etree.ElementTree as ET
    meta = meta or {}

    g = gpxpy.gpx.GPX()
    g.name         = meta.get("name")
    g.description  = meta.get("description")
    g.author_name  = meta.get("author_name")
    g.author_email = meta.get("author_email")
    g.link         = meta.get("link")
    g.link_text    = meta.get("link_text")
    g.time         = _parse_iso(meta.get("time"))
    g.keywords     = meta.get("keywords")

    trk = gpxpy.gpx.GPXTrack()
    trk.name = meta.get("track_name")
    trk.type = meta.get("track_type")
    g.tracks.append(trk)
    seg = gpxpy.gpx.GPXTrackSegment()
    trk.segments.append(seg)

    for rp in track:
        pt = gpxpy.gpx.GPXTrackPoint(
            latitude=rp.get("lat"),
            longitude=rp.get("lon"),
            elevation=rp.get("ele"),
            time=_parse_iso(rp.get("time")),
        )
        if rp.get("ext_xml"):
            try:
                wrapped = ET.fromstring(f"<root>{rp['ext_xml']}</root>")
                for child in wrapped:
                    pt.extensions.append(child)
            except Exception:
                pass
        seg.points.append(pt)

    return g.to_xml()


def track_to_json(result_points):
    """Convert internal result_points (datetime times) to JSON-safe track dicts."""
    return [
        {
            "lat": p["lat"], "lon": p["lon"], "ele": p.get("ele"),
            "time": _iso(p.get("time")), "ext_xml": p.get("ext_xml"),
        }
        for p in result_points
    ]


# ---------------------------------------------------------------------------
# Background processing worker
# ---------------------------------------------------------------------------

DEVIATION_THRESHOLD = 25   # metres — matches frontend


def process_job(job_id, gpx_bytes):
    job = jobs[job_id]
    try:
        job["status"] = "parsing"
        gpx_obj, original_points = parse_gpx(gpx_bytes)

        if not original_points:
            job["status"] = "error"
            job["error"] = "No trackpoints found in GPX file."
            return

        job["total_points"] = len(original_points)
        job["original_points"] = [{"lat": p["lat"], "lon": p["lon"]} for p in original_points]

        orig_coords = [(p["lat"], p["lon"]) for p in original_points]
        job["total_distance_m"] = total_distance_m(orig_coords)

        # Segmentation is done entirely client-side (junction-aware).
        # Backend just exposes the raw points and waits for the frontend to POST results.
        job["total_segments"] = None
        job["processed_segments"] = 0
        job["segments"] = []   # empty — frontend rebuilds from original_points
        job["status"] = "waiting_for_snapping"
        job["client_segments"] = None
        job["original_points_data"] = original_points
        job["gpx_obj"] = gpx_obj

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


def finalise_job(job_id):
    """Called when frontend posts all snapped segment results."""
    job = jobs[job_id]
    try:
        original_points = job["original_points_data"]

        # New payload format: segments list with explicit start_idx/end_idx/result
        data_segments = job.get("client_segments")  # [{start_idx, end_idx, result}]

        if not data_segments:
            # Fallback: old format
            snapped_segments = job["snapped_segments"]
            server_segments  = job["segments"]
            data_segments = [
                {
                    "start_idx": s["start_idx"],
                    "end_idx":   s["end_idx"],
                    "result":    snapped_segments.get(str(i)),
                }
                for i, s in enumerate(server_segments)
            ]

        result_points = []
        segment_map   = []

        for seg_i, seg in enumerate(data_segments):
            start   = seg["start_idx"]
            end     = seg["end_idx"]
            snapped = seg.get("result")  # list of {lat,lon} or None

            seg_original = original_points[start: end + 1]

            if snapped:
                max_dev = 0.0
                for sp in snapped:
                    min_d = min(
                        haversine(sp["lat"], sp["lon"], op["lat"], op["lon"])
                        for op in seg_original
                    )
                    max_dev = max(max_dev, min_d)
                use_snapped = max_dev <= DEVIATION_THRESHOLD
            else:
                use_snapped = False

            # Skip duplicate leading point for all segments after the first
            range_original = seg_original if seg_i == 0 else seg_original[1:]

            if use_snapped:
                pts_to_use = snapped if seg_i == 0 else snapped[1:]
                for k, sp in enumerate(pts_to_use):
                    orig_idx = start + (0 if seg_i == 0 else 1) + k
                    orig = original_points[min(orig_idx, len(original_points) - 1)]
                    result_points.append({
                        "lat":     sp["lat"],
                        "lon":     sp["lon"],
                        "ele":     orig["ele"],
                        "time":    orig["time"],
                        "ext_xml": orig.get("ext_xml"),
                    })
                    segment_map.append("snapped")
            else:
                for orig in range_original:
                    result_points.append(orig)
                    segment_map.append("original")

        total_pts    = len(segment_map)
        snapped_pts  = segment_map.count("snapped")
        unsnapped_pts = total_pts - snapped_pts

        result_coords = [(p["lat"], p["lon"]) for p in result_points]
        result_dist   = total_distance_m(result_coords)

        gpx_xml = build_gpx(job["gpx_obj"], result_points)

        job["result_gpx"]         = gpx_xml
        job["result_full"]        = result_points  # full dicts (kept for legacy routes)
        job["result_track"]       = track_to_json(result_points)  # JSON-safe, sent to frontend
        job["meta"]               = extract_meta(job["gpx_obj"])
        job["result_points"]      = [{"lat": p["lat"], "lon": p["lon"]} for p in result_points]
        job["result_points_ele"]  = [p.get("ele") for p in result_points]
        job["segment_map"]        = segment_map
        job["summary"] = {
            "total_points":       total_pts,
            "snapped_points":     snapped_pts,
            "unsnapped_points":   unsnapped_pts,
            "snapped_pct":        round(100 * snapped_pts / total_pts, 1) if total_pts else 0,
            "unsnapped_pct":      round(100 * unsnapped_pts / total_pts, 1) if total_pts else 0,
            "total_distance_m":   result_dist,
            "original_distance_m": job.get("total_distance_m", 0),
        }
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    gpx_bytes = f.read()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "created": datetime.now(timezone.utc).isoformat(),
    }
    t = threading.Thread(target=process_job, args=(job_id, gpx_bytes), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    resp = {
        "status": job["status"],
        "total_points": job.get("total_points"),
        "total_segments": job.get("total_segments"),
        "processed_segments": job.get("processed_segments"),
    }

    if job["status"] == "waiting_for_snapping":
        resp["original_points"] = job.get("original_points", [])
        resp["total_distance_m"] = job.get("total_distance_m")

    if job["status"] == "done":
        resp["result_points"] = job.get("result_points", [])
        resp["result_points_ele"] = job.get("result_points_ele", [])
        resp["segment_map"] = job.get("segment_map", [])
        resp["summary"] = job.get("summary", {})
        resp["original_points"] = job.get("original_points", [])
        resp["result_track"] = job.get("result_track", [])  # full track: frontend's source of truth
        resp["meta"] = job.get("meta", {})

    if job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")

    return jsonify(resp)


@app.route("/submit_snapped/<job_id>", methods=["POST"])
def submit_snapped(job_id):
    """Frontend POSTs all snapped segment results here."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    data = request.get_json()
    # New format: segments list with explicit indices and results
    if "segments" in data:
        job["client_segments"] = data["segments"]
    else:
        # Legacy fallback
        job["snapped_segments"] = data.get("snapped_segments", {})
    t = threading.Thread(target=finalise_job, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/submit_elevation/<job_id>", methods=["POST"])
def submit_elevation(job_id):
    """Frontend POSTs corrected elevation array; backend rebuilds the GPX."""
    job = jobs.get(job_id)
    if not job or "result_full" not in job:
        return jsonify({"error": "Result not ready"}), 404

    ele = (request.get_json() or {}).get("ele", [])
    pts = job["result_full"]
    for i, p in enumerate(pts):
        if i < len(ele) and ele[i] is not None:
            p["ele"] = ele[i]

    job["result_gpx"]        = build_gpx(job["gpx_obj"], pts)
    job["result_points_ele"] = [p.get("ele") for p in pts]
    return jsonify({"ok": True})


@app.route("/submit_splice/<job_id>", methods=["POST"])
def submit_splice(job_id):
    """Replace straight-line dropout runs with road-routed geometry.

    Each splice: {start, end, points:[{lat,lon}]} where start/end index into
    the current result_full array. Inserted points get ele/time linearly
    interpolated between the two endpoints by distance fraction.
    """
    job = jobs.get(job_id)
    if not job or "result_full" not in job:
        return jsonify({"error": "Result not ready"}), 404

    splices = (request.get_json() or {}).get("splices", [])
    pts = job["result_full"]
    seg = job["segment_map"]

    # Apply highest-index first so earlier indices stay valid
    splices.sort(key=lambda s: s["start"], reverse=True)

    for sp in splices:
        s, e = sp["start"], sp["end"]
        geo  = sp.get("points", [])
        if e <= s or len(geo) < 2 or e >= len(pts):
            continue

        a, b = pts[s], pts[e]

        # Cumulative distance along the new geometry → fraction for interpolation
        cum = [0.0]
        for i in range(1, len(geo)):
            cum.append(cum[-1] + haversine(geo[i-1]["lat"], geo[i-1]["lon"], geo[i]["lat"], geo[i]["lon"]))
        total = cum[-1] or 1.0

        def lerp(va, vb, f):
            if va is None or vb is None:
                return va if va is not None else vb
            return va + (vb - va) * f

        a_t = a.get("time"); b_t = b.get("time")
        new_pts = []
        for i, g in enumerate(geo):
            f = cum[i] / total
            t = None
            if a_t is not None and b_t is not None:
                t = a_t + (b_t - a_t) * f
            new_pts.append({
                "lat": g["lat"], "lon": g["lon"],
                "ele": lerp(a.get("ele"), b.get("ele"), f),
                "time": t,
                "ext_xml": None,
            })

        pts[s:e+1] = new_pts
        seg[s:e+1] = ["spliced"] * len(new_pts)

    # Rebuild everything
    job["result_full"]       = pts
    job["segment_map"]       = seg
    job["result_gpx"]        = build_gpx(job["gpx_obj"], pts)
    job["result_points"]     = [{"lat": p["lat"], "lon": p["lon"]} for p in pts]
    job["result_points_ele"] = [p.get("ele") for p in pts]

    total_pts   = len(seg)
    snapped_pts = sum(1 for t in seg if t in ("snapped", "spliced"))
    result_dist = total_distance_m([(p["lat"], p["lon"]) for p in pts])
    job["summary"] = {
        "total_points":        total_pts,
        "snapped_points":      snapped_pts,
        "unsnapped_points":    total_pts - snapped_pts,
        "snapped_pct":         round(100 * snapped_pts / total_pts, 1) if total_pts else 0,
        "unsnapped_pct":       round(100 * (total_pts - snapped_pts) / total_pts, 1) if total_pts else 0,
        "total_distance_m":    result_dist,
        "original_distance_m": job.get("total_distance_m", 0),
    }

    return jsonify({
        "ok": True,
        "result_points":     job["result_points"],
        "result_points_ele": job["result_points_ele"],
        "segment_map":       seg,
        "summary":           job["summary"],
    })


@app.route("/splice", methods=["POST"])
def splice():
    """Stateless splice. Body: { track:[...], segment_map:[...], splices:[{start,end,points}] }.
    Returns the updated track + derived arrays. Holds no server state.
    """
    data = request.get_json() or {}
    track = data.get("track", [])
    seg   = data.get("segment_map", [])
    splices = data.get("splices", [])
    if not track:
        return jsonify({"error": "No track provided"}), 400
    if len(seg) != len(track):
        seg = ["original"] * len(track)

    splices.sort(key=lambda s: s["start"], reverse=True)
    for sp in splices:
        s, e = sp["start"], sp["end"]
        geo  = sp.get("points", [])
        if e <= s or len(geo) < 2 or e >= len(track):
            continue
        a, b = track[s], track[e]

        cum = [0.0]
        for i in range(1, len(geo)):
            cum.append(cum[-1] + haversine(geo[i-1]["lat"], geo[i-1]["lon"], geo[i]["lat"], geo[i]["lon"]))
        total = cum[-1] or 1.0

        a_t = _parse_iso(a.get("time")); b_t = _parse_iso(b.get("time"))
        a_e = a.get("ele"); b_e = b.get("ele")

        new_pts = []
        for i, g in enumerate(geo):
            f = cum[i] / total
            t = None
            if a_t is not None and b_t is not None:
                t = _iso(a_t + (b_t - a_t) * f)
            ele = None
            if a_e is not None and b_e is not None:
                ele = a_e + (b_e - a_e) * f
            elif a_e is not None:
                ele = a_e
            new_pts.append({"lat": g["lat"], "lon": g["lon"], "ele": ele, "time": t, "ext_xml": None})

        track[s:e+1] = new_pts
        seg[s:e+1]   = ["spliced"] * len(new_pts)

    total_pts   = len(seg)
    snapped_pts = sum(1 for t in seg if t in ("snapped", "spliced"))
    result_dist = total_distance_m([(p["lat"], p["lon"]) for p in track])
    summary = {
        "total_points":        total_pts,
        "snapped_points":      snapped_pts,
        "unsnapped_points":    total_pts - snapped_pts,
        "snapped_pct":         round(100 * snapped_pts / total_pts, 1) if total_pts else 0,
        "unsnapped_pct":       round(100 * (total_pts - snapped_pts) / total_pts, 1) if total_pts else 0,
        "total_distance_m":    result_dist,
        "original_distance_m": data.get("original_distance_m", result_dist),
    }

    return jsonify({
        "ok": True,
        "track":             track,
        "segment_map":       seg,
        "result_points":     [{"lat": p["lat"], "lon": p["lon"]} for p in track],
        "result_points_ele": [p.get("ele") for p in track],
        "summary":           summary,
    })


@app.route("/build_gpx", methods=["POST"])
def build_gpx_route():
    """Stateless GPX builder. Body: { track:[...], meta:{...} } → GPX file download."""
    data  = request.get_json() or {}
    track = data.get("track", [])
    meta  = data.get("meta", {})
    if not track:
        return jsonify({"error": "No track provided"}), 400

    gpx_xml = build_gpx_stateless(track, meta)
    buf = io.BytesIO(gpx_xml.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, mimetype="application/gpx+xml", as_attachment=True, download_name="snapped.gpx")


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 404
    gpx_xml = job["result_gpx"]
    buf = io.BytesIO(gpx_xml.encode("utf-8") if isinstance(gpx_xml, str) else gpx_xml)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/gpx+xml",
        as_attachment=True,
        download_name="snapped.gpx",
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
