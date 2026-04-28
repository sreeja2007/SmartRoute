"""
route_engine.py
---------------
Smart Supply Chain - Route Optimization Engine
Runs on port 5001 (Flask), separate from the main app on port 5000.

APIs:
  POST /optimize-routes  -> KMeans clustering + Nearest-Neighbour TSP
  GET  /risk-score       -> IsolationForest ML risk scorer
  POST /reroute          -> Reroute a blocked vehicle
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import math
import networkx as nx
import threading
import requests
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime

try:
    import osmnx as ox
except Exception:
    ox = None

app = Flask(__name__)
CORS(app)

# ─── Warehouse / depot coordinates (Tamil Nadu - Chennai) ────────────────────
WAREHOUSE = {"lat": 13.0827, "lng": 80.2707}
_GRAPH_CACHE = {}
_GRAPH_LOCK = threading.Lock()

# ─── IsolationForest model (trained once at startup) ─────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "risk_model.pkl")

def train_risk_model():
    """
    Train IsolationForest on normal delivery conditions.
    Features: [rain_mm, wind_speed_ms, visibility_km, traffic_delay_min, temperature_c]
    Normal = low rain, low wind, good visibility, low delay, moderate temp.
    """
    rng = np.random.default_rng(42)
    n = 500

    normal_data = np.column_stack([
        rng.uniform(0, 2, n),       # rain_mm       : 0-2 = normal
        rng.uniform(0, 8, n),       # wind_speed_ms  : 0-8 = normal
        rng.uniform(5, 10, n),      # visibility_km  : 5-10 = normal
        rng.uniform(0, 10, n),      # traffic_delay_min : 0-10 = normal
        rng.uniform(15, 35, n),     # temperature_c  : 15-35 = normal
    ])

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,   # 10% of training data treated as anomalies
        random_state=42
    )
    model.fit(normal_data)
    joblib.dump(model, MODEL_PATH)
    print("Risk model trained and saved.")
    return model

# Load or train model at startup
if os.path.exists(MODEL_PATH):
    risk_model = joblib.load(MODEL_PATH)
    print("Risk model loaded from disk.")
else:
    risk_model = train_risk_model()


# ─── Helper: Euclidean distance between two lat/lng points (in km) ───────────
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    from math import radians, sin, cos, sqrt, atan2
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─── Job 1: KMeans clustering ─────────────────────────────────────────────────
def kmeans_cluster(orders, num_vehicles):
    """Split orders into num_vehicles groups using KMeans."""
    if len(orders) <= num_vehicles:
        return [[o] for o in orders]

    coords = np.array([[o["lat"], o["lng"]] for o in orders])
    k = min(num_vehicles, len(orders))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)

    clusters = [[] for _ in range(k)]
    for order, label in zip(orders, labels):
        clusters[label].append(order)

    return [c for c in clusters if c]


# ─── Job 2: Nearest-Neighbour TSP (simple greedy, no NetworkX overhead) ──────
def nearest_neighbour_tsp(stops, start):
    if not stops:
        return []
    remaining = list(stops)
    ordered = []
    current = start
    while remaining:
        nearest = min(remaining, key=lambda s: haversine(current["lat"], current["lng"], s["lat"], s["lng"]))
        ordered.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return ordered


def _interpolate(p1, p2, steps=20):
    """Return `steps` intermediate [lat, lng] points between p1 and p2."""
    pts = []
    for s in range(steps + 1):
        t = s / steps
        pts.append([p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])])
    return pts


def _path_distance_km(coords):
    if not coords or len(coords) < 2:
        return 0.0
    return sum(
        haversine(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        for i in range(len(coords) - 1)
    )


def _graph_cache_key(points):
    lats = [p["lat"] for p in points]
    lngs = [p["lng"] for p in points]
    return (
        round(min(lats) - 0.2, 2),
        round(max(lats) + 0.2, 2),
        round(min(lngs) - 0.2, 2),
        round(max(lngs) + 0.2, 2),
    )


def _load_graph_for_points(points):
    if ox is None:
        return None

    key = _graph_cache_key(points)
    with _GRAPH_LOCK:
        if key in _GRAPH_CACHE:
            return _GRAPH_CACHE[key]

    south, north, west, east = key
    # OSMnx has multiple API shapes across versions.
    # - 2.x supports bbox=(left, bottom, right, top)
    # - 1.x commonly uses north, south, east, west positional/keyword args
    try:
        sig = inspect.signature(ox.graph_from_bbox)
        if "bbox" in sig.parameters:
            bbox = (west, south, east, north)  # (left, bottom, right, top)
            G = ox.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True)
        else:
            G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="drive", simplify=True)
    except TypeError:
        # Extra safety: if signature inspection lies (wrapped functions), try both.
        try:
            bbox = (west, south, east, north)
            G = ox.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True)
        except Exception:
            G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="drive", simplify=True)

    with _GRAPH_LOCK:
        _GRAPH_CACHE[key] = G
    return G


def _road_segment(G, start, end):
    if G is None or ox is None:
        return None
    try:
        start_node = ox.distance.nearest_nodes(G, start["lng"], start["lat"])
        end_node = ox.distance.nearest_nodes(G, end["lng"], end["lat"])
        path = nx.shortest_path(G, start_node, end_node, weight="length", method="dijkstra")
        return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in path]
    except Exception:
        return None


def _osrm_segment(start, end, timeout_s=10):
    """
    Query OSRM for a road-following path between two points.
    Uses OSRM's `route` service with GeoJSON geometry.
    Returns list of [lat, lng] points or None.
    """
    base = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
    url = (
        f"{base}/route/v1/driving/"
        f"{start['lng']},{start['lat']};{end['lng']},{end['lat']}"
        f"?overview=full&geometries=geojson&steps=false"
    )
    try:
        resp = requests.get(url, timeout=float(os.getenv("OSRM_TIMEOUT_S", timeout_s)))
        if resp.status_code != 200:
            return None
        data = resp.json()
        routes = data.get("routes") or []
        if not routes:
            return None
        geom = routes[0].get("geometry") or {}
        coords = geom.get("coordinates") or []
        if not coords:
            return None
        # OSRM returns [lng, lat] → convert to [lat, lng]
        return [[lat, lng] for (lng, lat) in coords]
    except Exception:
        return None


def _load_graph_for_points_with_timeout(points, timeout_s=20.0):
    """
    OSMnx graph downloads can hang (Overpass/network). Load in a worker thread with timeout.
    Returns (G, error_str). On timeout/failure, G is None and error_str explains why.
    """
    if ox is None:
        return None, "osmnx_not_available"

    def _load():
        return _load_graph_for_points(points)

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_load)
            G = fut.result(timeout=float(os.getenv("OSMNX_TIMEOUT_S", timeout_s)))
            return G, None
    except FuturesTimeoutError:
        return None, "osmnx_graph_load_timeout"
    except Exception as e:
        return None, f"osmnx_graph_load_failed:{e}"


def build_route_coordinates(waypoints, allow_osmnx_fallback=True):
    """
    Build road-following geometry between waypoints.
    Priority:
      1) OSRM (fast, road-following, no local graph)
      2) OSMnx + NetworkX (offline-ish, but heavy)
      3) Interpolation fallback (straight line)
    """
    points = [{"lat": w["lat"], "lng": w["lng"]} for w in waypoints]
    G = None
    graph_error = None

    route_coordinates = []
    debug = {
        "segments": max(0, len(waypoints) - 1),
        "osrm_segments": 0,
        "osmnx_segments": 0,
        "interpolated_segments": 0,
        "osmnx_graph_loaded": False,
        "osmnx_graph_error": None,
        "osrm_base_url": os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org"),
    }
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]

        # 1) OSRM road geometry
        segment = _osrm_segment(start, end)
        if segment:
            debug["osrm_segments"] += 1

        # 2) OSMnx fallback (optional; can be disabled for low-latency reroutes)
        if not segment and allow_osmnx_fallback:
            # Lazy-load graph only when needed, and with timeout.
            if G is None and graph_error is None:
                G, graph_error = _load_graph_for_points_with_timeout(points)
                debug["osmnx_graph_loaded"] = G is not None
                debug["osmnx_graph_error"] = graph_error
                if graph_error:
                    print(f"OSMnx graph unavailable, will skip OSMnx routing. Reason: {graph_error}")

            segment = _road_segment(G, start, end)
            if segment:
                debug["osmnx_segments"] += 1

        # 3) Straight-line fallback
        if not segment:
            p1 = [start["lat"], start["lng"]]
            p2 = [end["lat"], end["lng"]]
            seg_km = haversine(p1[0], p1[1], p2[0], p2[1])
            steps = max(10, int(seg_km * 20))
            segment = _interpolate(p1, p2, steps)
            debug["interpolated_segments"] += 1

        if i == 0:
            route_coordinates.extend(segment)
        else:
            route_coordinates.extend(segment[1:])
    if debug["interpolated_segments"] == debug["segments"]:
        routing_method = "Interpolation (Straight Line)"
    elif debug["osmnx_segments"] > 0 and debug["osrm_segments"] == 0:
        routing_method = "OSMnx + NetworkX"
    elif debug["osrm_segments"] > 0:
        routing_method = "OSRM"
    else:
        routing_method = "Mixed"

    return route_coordinates, routing_method, debug


def build_route_for_vehicle(vehicle_id, cluster_orders):
    """Run TSP on a cluster and return road-following route geometry."""
    ordered = nearest_neighbour_tsp(cluster_orders, WAREHOUSE)

    waypoints = [WAREHOUSE] + ordered + [WAREHOUSE]
    route_coordinates, routing_method, routing_debug = build_route_coordinates(waypoints)
    total_km = _path_distance_km(route_coordinates)

    avg_speed_kmh = 25
    eta_minutes = int((total_km / avg_speed_kmh) * 60) + len(ordered) * 2

    return {
        "vehicle_id": vehicle_id,
        "assigned_orders": ordered,
        "route_coordinates": route_coordinates,
        "total_distance_km": round(total_km, 2),
        "estimated_time_min": eta_minutes,
        "stop_count": len(ordered),
        "routing_method": routing_method,
        "routing_debug": routing_debug,
    }


# ─── Job 3: IsolationForest risk scorer ──────────────────────────────────────
def compute_risk_score(features):
    """
    features: [rain_mm, wind_speed_ms, visibility_km, traffic_delay_min, temperature_c]
    Returns risk_score in [0, 1].
    Combines IsolationForest anomaly signal with direct feature thresholds
    so that clearly bad conditions always score high.
    """
    rain, wind, visibility, delay, temperature = features

    # IsolationForest anomaly component (0-1, higher = more anomalous)
    X = np.array([features])
    raw = risk_model.decision_function(X)[0]
    # raw is typically -0.15 to +0.15; normalise to 0-1
    iso_component = float(np.clip(0.5 - raw * 3.0, 0.0, 1.0))

    # Rule-based component from individual features
    rain_risk      = min(1.0, rain / 20.0)           # 20mm = max risk
    wind_risk      = min(1.0, wind / 20.0)            # 20 m/s = max risk
    vis_risk       = min(1.0, max(0.0, (5.0 - visibility) / 5.0))  # <5km = risk
    delay_risk     = min(1.0, delay / 60.0)           # 60 min = max risk

    rule_component = (rain_risk * 0.35 + wind_risk * 0.25 +
                      vis_risk * 0.25 + delay_risk * 0.15)

    # Blend: 40% IsolationForest + 60% rule-based
    risk = round(float(np.clip(0.4 * iso_component + 0.6 * rule_component, 0.0, 1.0)), 3)

    if risk >= 0.80:
        severity = "critical"
    elif risk >= 0.65:
        severity = "high"
    elif risk >= 0.40:
        severity = "moderate"
    else:
        severity = "low"

    return {
        "risk_score": risk,
        "severity": severity,
        "is_disruption": risk >= 0.65,
        "recommendation": (
            "IMMEDIATE REROUTE REQUIRED" if risk >= 0.80 else
            "REROUTE RECOMMENDED" if risk >= 0.65 else
            "MONITOR CLOSELY" if risk >= 0.40 else
            "PROCEED AS PLANNED"
        )
    }


# ─── API 1: POST /optimize-routes ────────────────────────────────────────────
@app.route("/optimize-routes", methods=["POST"])
def optimize_routes():
    """
    Body:
      {
        "orders": [{"id": "ORD1", "lat": 28.61, "lng": 77.20, ...}, ...],
        "vehicle_count": 3
      }
    Returns best route per vehicle.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON body"}), 400

        orders = data.get("orders", [])
        vehicle_count = int(data.get("vehicle_count", 3))

        if not orders:
            return jsonify({"success": False, "error": "No orders provided"}), 400

        if vehicle_count < 1:
            return jsonify({"success": False, "error": "vehicle_count must be >= 1"}), 400

        # Step 1 — KMeans clustering
        clusters = kmeans_cluster(orders, vehicle_count)

        # Step 2 — Nearest-neighbour TSP per cluster
        routes = []
        total_distance = 0.0
        for i, cluster in enumerate(clusters):
            vehicle_id = f"V{i+1}"
            route = build_route_for_vehicle(vehicle_id, cluster)
            routes.append(route)
            total_distance += route["total_distance_km"]

        naive_distance = len(orders) * 2.0
        distance_saved = max(0, naive_distance - total_distance)
        efficiency_pct = round((distance_saved / naive_distance * 100) if naive_distance > 0 else 0, 1)

        return jsonify({
            "success": True,
            "routes": routes,
            "summary": {
                "total_orders": len(orders),
                "vehicle_count": len(routes),
                "total_distance_km": round(total_distance, 2),
                "distance_saved_km": round(distance_saved, 2),
                "efficiency_pct": efficiency_pct,
                "algorithm": "KMeans + Nearest-Neighbour TSP (NetworkX)"
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── API 2: GET /risk-score ───────────────────────────────────────────────────
@app.route("/risk-score", methods=["GET"])
def risk_score():
    """
    Query params:
      rain_mm          (float, default 0)
      wind_speed_ms    (float, default 0)
      visibility_km    (float, default 10)
      traffic_delay_min(float, default 0)
      temperature_c    (float, default 25)

    Returns: { risk_score, severity, is_disruption, recommendation }
    """
    try:
        rain        = float(request.args.get("rain_mm", 0))
        wind        = float(request.args.get("wind_speed_ms", 0))
        visibility  = float(request.args.get("visibility_km", 10))
        delay       = float(request.args.get("traffic_delay_min", 0))
        temperature = float(request.args.get("temperature_c", 25))

        features = [rain, wind, visibility, delay, temperature]
        result = compute_risk_score(features)

        return jsonify({
            "success": True,
            "input": {
                "rain_mm": rain,
                "wind_speed_ms": wind,
                "visibility_km": visibility,
                "traffic_delay_min": delay,
                "temperature_c": temperature
            },
            **result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── API 3: POST /reroute ─────────────────────────────────────────────────────
@app.route("/reroute", methods=["POST"])
def reroute():
    """
    Body:
      {
        "vehicle_id": "V1",
        "current_position": {"lat": 28.61, "lng": 77.20},
        "remaining_orders": [{"id": "ORD5", "lat": ..., "lng": ...}, ...],
        "reason": "storm detected"
      }
    Returns a new optimised route from current position through remaining stops.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON body"}), 400

        vehicle_id       = data.get("vehicle_id", "V1")
        current_position = data.get("current_position", WAREHOUSE)
        remaining_orders = data.get("remaining_orders", [])
        reason           = data.get("reason", "disruption detected")

        if not remaining_orders:
            return jsonify({
                "success": True,
                "vehicle_id": vehicle_id,
                "message": "No remaining orders — route complete",
                "new_route": [],
                "reason": reason
            })

        # Re-run nearest-neighbour TSP from current position
        ordered = nearest_neighbour_tsp(remaining_orders, current_position)

        waypoints = [current_position] + ordered + [WAREHOUSE]
        route_coordinates = build_route_coordinates(waypoints)
        total_km = _path_distance_km(route_coordinates)

        avg_speed_kmh = 25
        new_eta_minutes = int((total_km / avg_speed_kmh) * 60) + len(ordered) * 2

        return jsonify({
            "success": True,
            "vehicle_id": vehicle_id,
            "reason": reason,
            "new_route": {
                "assigned_orders": ordered,
                "route_coordinates": route_coordinates,
                "total_distance_km": round(total_km, 2),
                "new_eta_minutes": new_eta_minutes,
                "stop_count": len(ordered)
            },
            "rerouted_at": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Smart Supply Chain Route Engine",
        "model_loaded": risk_model is not None,
        "timestamp": datetime.now().isoformat()
    })


def run_optimization(orders, vehicle_count=3):
    """Importable by FastAPI — no Flask context needed."""
    clusters = kmeans_cluster(orders, vehicle_count)

    routes = []
    total_distance = 0.0

    for i, cluster in enumerate(clusters):
        vehicle_id = f"V{i+1}"
        route = build_route_for_vehicle(vehicle_id, cluster)
        routes.append(route)
        total_distance += route["total_distance_km"]

    return {
        "routes": routes,
        "total_distance_km": round(total_distance, 2)
    }


if __name__ == "__main__":
    print("=== ROUTE ENGINE STARTING ON PORT 5001 ===")
    app.run(debug=True, host="0.0.0.0", port=5001)