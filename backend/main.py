from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import io
import shutil
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from typing import Any
from sqlalchemy import or_
from datetime import datetime

from modules.weather_service import get_weather, calculate_weather_score
from modules.route_manager import RouteManager
from modules.route_engine import run_optimization, compute_risk_score, haversine, WAREHOUSE, build_route_coordinates

from database import SessionLocal, engine
from models import Base, Order
from schemas import OrderCreate
from geocoder import geocode, API_KEY
from parser import parse_pdf, parse_docx

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Start with a clean slate; orders come only from uploaded CSV.
        db.query(Order).delete()
        db.commit()
    finally:
        db.close()
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


# ---------------- DB CONNECTION ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- GLOBAL ROUTE MANAGER ----------------
route_manager = RouteManager()

# ---------------- SIMULATION STATE (in-memory) ----------------
simulation_state = {
    "active_routes": [],  # List of {vehicle_id, route_coordinates, current_index}
    "fleet_deployed": False,
    "reroute_events": [],
    "route_optimized": False,
    "multi_vehicle_routes": [],
    "active_vehicles": [],
    "route_history": {},  # vehicle_id -> list of route versions/events
    "paused_vehicles": [],
    "dispatcher_events": [],
}


def _is_vehicle_paused(vehicle_id: str) -> bool:
    return vehicle_id in (simulation_state.get("paused_vehicles") or [])


def _set_vehicle_paused(vehicle_id: str, paused: bool, reason: str = "") -> None:
    paused_list = list(simulation_state.get("paused_vehicles") or [])
    if paused and vehicle_id not in paused_list:
        paused_list.append(vehicle_id)
    if (not paused) and vehicle_id in paused_list:
        paused_list.remove(vehicle_id)
    simulation_state["paused_vehicles"] = paused_list
    simulation_state["dispatcher_events"].append({
        "type": "pause" if paused else "resume",
        "vehicle_id": vehicle_id,
        "reason": reason or ("Paused by dispatcher" if paused else "Resumed by dispatcher"),
        "timestamp": datetime.now().isoformat(),
    })


def _path_distance_km(coords: list[list[float]]) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return float(total)


def _min_distance_m_to_polyline(lat: float, lng: float, coords: list[list[float]], start_idx: int = 0, window: int = 120) -> tuple[float, int]:
    """
    Lightweight deviation detector: find nearest polyline point distance (meters) in a window.
    Returns (min_distance_m, nearest_index).
    """
    if not coords:
        return float("inf"), 0
    s = max(0, int(start_idx) - int(window // 2))
    e = min(len(coords), int(start_idx) + int(window // 2) + 1)
    if s >= e:
        s, e = 0, len(coords)

    best_m = float("inf")
    best_i = s
    for i in range(s, e):
        d_km = haversine(lat, lng, coords[i][0], coords[i][1])
        d_m = d_km * 1000.0
        if d_m < best_m:
            best_m = d_m
            best_i = i
    return best_m, best_i


def _reroute_preserve_sequence(vehicle_id: str, current_position: dict[str, float], reason: str, db: Session) -> dict[str, Any]:
    """
    Recompute route geometry from current position through remaining stops in the SAME order.
    Only path/timing changes; stop list + order sequence is preserved.
    """
    # Find current assigned order sequence for this vehicle.
    route_entry = next((r for r in (simulation_state.get("multi_vehicle_routes") or []) if r.get("vehicle_id") == vehicle_id), None)
    assigned = (route_entry or {}).get("assigned_orders") or []

    # Filter to remaining (not delivered) using DB as source of truth.
    remaining: list[dict[str, float]] = []
    for o in assigned:
        oid = o.get("id")
        if oid is None:
            continue
        order_db = db.query(Order).filter(Order.id == int(oid)).first()
        if order_db and (order_db.status or "pending") == "delivered":
            continue
        remaining.append({"id": str(oid), "lat": float(o["lat"]), "lng": float(o["lng"])})

    if not remaining:
        return {"success": True, "vehicle_id": vehicle_id, "message": "No remaining stops", "rerouted": False}

    waypoints = [{"lat": float(current_position["lat"]), "lng": float(current_position["lng"])}] + remaining + [WAREHOUSE]
    # Manual/deviation reroutes should be low-latency for operator UX.
    # Prefer OSRM + interpolation fallback; skip OSMnx heavy fallback here.
    route_coordinates, routing_method, routing_debug = build_route_coordinates(waypoints, allow_osmnx_fallback=False)
    total_km = _path_distance_km(route_coordinates)
    avg_speed_kmh = 25.0
    eta_minutes = int((total_km / avg_speed_kmh) * 60.0) + len(remaining) * 2

    # Update sim state: active_routes geometry + multi_vehicle_routes geometry.
    for ar in simulation_state.get("active_routes") or []:
        if ar.get("vehicle_id") == vehicle_id:
            ar["route_coordinates"] = route_coordinates
            ar["current_index"] = 0

    for mr in simulation_state.get("multi_vehicle_routes") or []:
        if mr.get("vehicle_id") == vehicle_id:
            mr["route_coordinates"] = route_coordinates
            mr["total_distance_km"] = round(total_km, 2)
            mr["estimated_time_min"] = eta_minutes
            mr["routing_method"] = routing_method
            mr["routing_debug"] = routing_debug
            mr["rerouted"] = True
            mr["reroute_reason"] = reason
            mr["reroute_timestamp"] = datetime.now().isoformat()

    # Record reroute event + history
    event = {
        "vehicle_id": vehicle_id,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "updated_eta_min": eta_minutes,
        "updated_distance_km": round(total_km, 2),
        "routing_method": routing_method,
    }
    simulation_state["reroute_events"].append(event)
    simulation_state["route_history"].setdefault(vehicle_id, []).append(event)

    return {
        "success": True,
        "vehicle_id": vehicle_id,
        "rerouted": True,
        "reason": reason,
        "updated_eta_min": eta_minutes,
        "updated_distance_km": round(total_km, 2),
        "route_coordinates": route_coordinates,
        "remaining_stops": remaining,
        "routing_method": routing_method,
        "routing_debug": routing_debug,
        "event": event,
    }

# ---------------- UI ----------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


# ---------------- UI BRIDGE APIs (called by admin.html) ----------------
@app.post("/api/optimize_route")
def api_optimize_route(body: dict = {}):
    vehicle_count = int(body.get("vehicle_count", 3))
    db = SessionLocal()
    try:
        # Treat NULL/empty status as pending so newly-uploaded rows are included.
        orders_db = db.query(Order).filter(or_(Order.status == "pending", Order.status.is_(None), Order.status == "")).all()
    finally:
        db.close()

    if not orders_db:
        raise HTTPException(status_code=400, detail="No pending orders to optimize")

    orders = [{"id": str(o.id), "lat": o.latitude, "lng": o.longitude} for o in orders_db]
    result = run_optimization(orders, vehicle_count)

    routes = result["routes"]
    active_vehicles = [r["vehicle_id"] for r in routes]
    total_distance = result["total_distance_km"]
    # Naive baseline: serve each stop individually (warehouse -> stop -> warehouse),
    # using straight-line haversine distance. This baseline is always comparable and
    # avoids showing 0% just because a placeholder baseline was too small.
    warehouse_lat = float(WAREHOUSE["lat"])
    warehouse_lng = float(WAREHOUSE["lng"])
    naive_distance = sum(
        2.0 * haversine(warehouse_lat, warehouse_lng, float(o["lat"]), float(o["lng"]))
        for o in orders
    )
    distance_saved = max(0.0, naive_distance - float(total_distance))
    overall_efficiency_pct = round((distance_saved / naive_distance * 100) if naive_distance > 0 else 0.0, 1)

    # Attach efficiency_pct and estimated_time_min if missing
    for r in routes:
        if "efficiency_pct" not in r:
            r["efficiency_pct"] = overall_efficiency_pct
        if "estimated_time_min" not in r:
            r["estimated_time_min"] = r.get("estimated_time_min", 0)
        r["orders_visited"] = r.get("stop_count", len(r.get("assigned_orders", [])))

    # Store routes in simulation state for weather monitoring
    simulation_state["active_routes"] = [
        {"vehicle_id": r["vehicle_id"], "route_coordinates": r["route_coordinates"], "current_index": 0}
        for r in routes
    ]
    simulation_state["fleet_deployed"] = False
    simulation_state["reroute_events"] = []
    simulation_state["route_optimized"] = True
    simulation_state["multi_vehicle_routes"] = routes
    simulation_state["active_vehicles"] = active_vehicles
    simulation_state["paused_vehicles"] = []

    return {
        "success": True,
        "multi_vehicle_routes": routes,
        "active_vehicles": active_vehicles,
        "total_distance_km": total_distance,
        "orders_visited": len(orders),
        "distance_saved_km": round(distance_saved, 2),
        "efficiency_pct": overall_efficiency_pct,
    }


@app.post("/api/spawn_truck")
def api_spawn_truck():
    if not simulation_state.get("route_optimized"):
        raise HTTPException(status_code=400, detail="No optimized routes available. Optimize routes first.")

    # If active_routes isn't populated for any reason, rebuild from stored optimized routes.
    if not simulation_state.get("active_routes"):
        rebuilt = []
        for r in (simulation_state.get("multi_vehicle_routes") or []):
            coords = r.get("route_coordinates") or []
            if not coords:
                continue
            rebuilt.append({"vehicle_id": r.get("vehicle_id"), "route_coordinates": coords, "current_index": 0})
        simulation_state["active_routes"] = rebuilt
        simulation_state["active_vehicles"] = [r.get("vehicle_id") for r in (simulation_state.get("multi_vehicle_routes") or []) if r.get("vehicle_id")]
        if not simulation_state["active_routes"]:
            raise HTTPException(status_code=400, detail="Optimized routes missing geometry. Re-optimize and try again.")

    simulation_state["fleet_deployed"] = True
    vehicle_positions: dict[str, Any] = {}
    for r in simulation_state["active_routes"]:
        vid = r["vehicle_id"]
        coords = r.get("route_coordinates") or []
        if not coords:
            continue
        vehicle_positions[vid] = {
            "lat": coords[0][0],
            "lng": coords[0][1],
            "current_index": 0,
            "total_road_points": len(coords),
        }
    return {
        "success": True,
        "vehicle_positions": vehicle_positions,
        "active_vehicles": simulation_state.get("active_vehicles", list(vehicle_positions.keys())),
        "message": "Fleet deployed — animation runs client-side"
    }


@app.post("/api/reset_simulation")
def api_reset_simulation():
    db = SessionLocal()
    try:
        db.query(Order).delete()
        db.commit()
    finally:
        db.close()
    simulation_state["active_routes"] = []
    simulation_state["fleet_deployed"] = False
    simulation_state["reroute_events"] = []
    simulation_state["route_optimized"] = False
    simulation_state["multi_vehicle_routes"] = []
    simulation_state["active_vehicles"] = []
    simulation_state["paused_vehicles"] = []
    simulation_state["dispatcher_events"] = []
    return {"success": True, "message": "Simulation reset and cleared"}


@app.get("/api/get_simulation_status")
def api_get_simulation_status():
    db = SessionLocal()
    try:
        all_orders = db.query(Order).all()
    finally:
        db.close()

    orders_list = [
        {
            "id": str(o.id), "lat": o.latitude, "lng": o.longitude,
            "type": "order", "delivery_status": o.status or "pending",
            "customer_name": o.customer_name, "address": o.address, "source": "db"
        }
        for o in all_orders
    ]
    pending = [o for o in orders_list if o["delivery_status"] == "pending"]
    delivered = [o for o in orders_list if o["delivery_status"] == "delivered"]

    return {
        "success": True,
        "simulation": {
            "orders": orders_list,
            "pending_orders": pending,
            "delivered_orders": delivered,
            "city_generated": len(orders_list) > 0,
            "route_optimized": bool(simulation_state.get("route_optimized")),
            "multi_vehicle_routes": simulation_state.get("multi_vehicle_routes", []),
            "active_vehicles": simulation_state.get("active_vehicles", []),
            "vehicle_positions": {
                r["vehicle_id"]: {
                    "lat": (r.get("route_coordinates") or [[None, None]])[r.get("current_index", 0)][0],
                    "lng": (r.get("route_coordinates") or [[None, None]])[r.get("current_index", 0)][1],
                    "current_index": r.get("current_index", 0),
                    "total_road_points": len(r.get("route_coordinates") or []),
                }
                for r in (simulation_state.get("active_routes") or [])
                if (r.get("route_coordinates") or [])
            },
            "vehicle_states": {},
            "reroute_events": simulation_state.get("reroute_events", []),
            "route_history": simulation_state.get("route_history", {}),
            "fleet_deployed": bool(simulation_state.get("fleet_deployed")),
            "paused_vehicles": simulation_state.get("paused_vehicles", []),
            "dispatcher_events": simulation_state.get("dispatcher_events", [])[-50:],
        }
    }


@app.post("/api/update-order-status")
def api_update_order_status(body: dict, db: Session = Depends(get_db)):
    order_id = body.get("id")
    status = body.get("status")
    if not order_id or not status:
        raise HTTPException(status_code=400, detail="id and status required")
    order = db.query(Order).filter(Order.id == int(order_id)).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order.status = status
    db.commit()
    return {"success": True}


@app.get("/api/weather-status")
async def api_weather_status(db: Session = Depends(get_db)):
    """Sample weather at each active vehicle's current position and return disruption alerts."""
    active_routes = simulation_state["active_routes"]
    if not active_routes:
        orders = db.query(Order).filter(Order.status == "pending").all()
        sample_points = [
            {"vehicle_id": f"sample_{o.id}", "lat": o.latitude, "lng": o.longitude}
            for o in orders[:5]
        ]
    else:
        sample_points = []
        for route in active_routes:
            idx = route["current_index"]
            coords = route["route_coordinates"]
            if coords and idx < len(coords):
                sample_points.append({
                    "vehicle_id": route["vehicle_id"],
                    "lat": coords[idx][0],
                    "lng": coords[idx][1]
                })

    if not sample_points:
        return {
            "success": True,
            "risk_assessment": {"risk_required": False, "risk_level": "LOW", "recommendation": "No active vehicles"},
            "route_impacts": [],
            "weather_snapshots": [],
            "reroute_events": simulation_state["reroute_events"],
            "total_deliveries": 0,
            "pending_deliveries": 0,
        }

    # Fetch weather for all sample points concurrently
    tasks = [get_weather(p["lat"], p["lng"]) for p in sample_points]
    weather_results = await asyncio.gather(*tasks)

    route_impacts = []
    weather_snapshots = []
    max_risk = 0.0
    for point, weather_data in zip(sample_points, weather_results):
        score = calculate_weather_score(weather_data)
        wind = weather_data.get("wind", {}).get("speed", 0)
        rain = weather_data.get("rain", {}).get("1h", 0)
        visibility = weather_data.get("visibility", 10000) / 1000
        temp = weather_data.get("main", {}).get("temp", 25)
        humidity = weather_data.get("main", {}).get("humidity", 0)
        condition = weather_data.get("weather", [{}])[0].get("description", "")
        delay = min(60, score * 0.6)
        features = [rain, wind, visibility, delay, temp]
        risk = compute_risk_score(features)
        max_risk = max(max_risk, risk["risk_score"])

        weather_snapshots.append({
            "vehicle_id": point["vehicle_id"],
            "lat": point["lat"],
            "lng": point["lng"],
            "condition": condition,
            "temp_c": temp,
            "humidity_pct": humidity,
            "wind_speed_ms": wind,
            "rain_mm": rain,
            "visibility_km": visibility,
            "risk_score": risk["risk_score"],
            "risk_severity": risk["severity"],
        })

        if risk["is_disruption"]:
            route_impacts.append({
                "vehicle_id": point["vehicle_id"],
                "risk_score": risk["risk_score"],
                "severity": risk["severity"],
                "recommendation": f"{point['vehicle_id']}: {risk['recommendation']} — {condition or 'severe conditions'}",
                "weather": {
                    "wind_speed": wind,
                    "rain_mm": rain,
                    "visibility_km": visibility,
                    "description": condition
                }
            })

            # Trigger reroute in simulation state
            for route in simulation_state["active_routes"]:
                if route["vehicle_id"] == point["vehicle_id"]:
                    event = {"vehicle_id": point["vehicle_id"], "reason": risk["recommendation"], "risk_score": risk["risk_score"]}
                    if event not in simulation_state["reroute_events"]:
                        simulation_state["reroute_events"].append(event)

    total = db.query(Order).count()
    pending = db.query(Order).filter(Order.status == "pending").count()

    risk_required = max_risk >= 0.65
    return {
        "success": True,
        "risk_assessment": {
            "risk_required": risk_required,
            "risk_level": "HIGH" if max_risk >= 0.65 else "MODERATE" if max_risk >= 0.40 else "LOW",
            "recommendation": "Rerouting recommended for affected vehicles" if risk_required else "All routes safe to proceed"
        },
        "route_impacts": route_impacts,
        "weather_snapshots": weather_snapshots,
        "reroute_events": simulation_state["reroute_events"],
        "total_deliveries": total,
        "pending_deliveries": pending
    }


@app.post("/api/step")
async def api_step(hours: float = 0.1):
    """Advance simulation: step route manager, check weather, trigger reroutes."""
    route_manager.step_all(hours)

    tasks = [get_weather(d.vehicle_location.lat, d.vehicle_location.lon) for d in route_manager.deliveries]
    weather_results = await asyncio.gather(*tasks)

    rerouted = []
    for i, delivery in enumerate(route_manager.deliveries):
        weather_data = weather_results[i]
        score = calculate_weather_score(weather_data)
        delivery.weather_info = weather_data
        delivery.weather_score = score
        if score > 50:
            delivery.trigger_reroute("Severe weather detected")
            rerouted.append(delivery.id)

    # Advance current_index for active simulation routes
    for route in simulation_state["active_routes"]:
        coords = route["route_coordinates"]
        if route["current_index"] < len(coords) - 1:
            route["current_index"] += max(1, int(len(coords) * hours * 0.1))

    return {
        "message": "Step complete",
        "rerouted_vehicles": rerouted,
        "deliveries": [d.get_state() for d in route_manager.deliveries]
    }


# ---------------- TRACKING APIs ----------------
@app.get("/deliveries")
def get_deliveries():
    return route_manager.get_all_states()


@app.post("/step")
async def step_simulation(hours: float = 0.1):
    route_manager.step_all(hours)

    tasks = []
    for delivery in route_manager.deliveries:
        lat = delivery.vehicle_location.lat
        lon = delivery.vehicle_location.lon
        tasks.append(get_weather(lat, lon))

    weather_results = await asyncio.gather(*tasks)

    for i, delivery in enumerate(route_manager.deliveries):
        weather_data = weather_results[i]
        score = calculate_weather_score(weather_data)

        delivery.weather_info = weather_data
        delivery.weather_score = score

        if score > 50:
            delivery.trigger_reroute("Severe weather detected")

    return {"message": "Simulation updated with weather"}

# ---------------- ORDER APIs ----------------
@app.get("/orders")
def get_orders(db: Session = Depends(get_db)):
    return db.query(Order).all()


@app.post("/manual-order")
def create_order(order: OrderCreate, db: Session = Depends(get_db)):
    lat, lng = geocode(order.address)

    if lat is None:
        raise HTTPException(status_code=400, detail="Invalid address")

    new_order = Order(
        customer_name=order.customer_name,
        address=order.address,
        latitude=lat,
        longitude=lng
    )

    db.add(new_order)
    db.commit()
    db.refresh(new_order)

    return new_order


@app.get("/orders/coordinates")
def get_coordinates(db: Session = Depends(get_db)):
    orders = db.query(Order).all()

    return [
        {"id": o.id, "lat": o.latitude, "lng": o.longitude}
        for o in orders
    ]


@app.patch("/orders/{order_id}/status")
def update_status(order_id: int, status: str, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.status = status
    db.commit()

    return {"message": "Status updated"}


@app.post("/assign-vehicle")
def assign_vehicle(order_id: int, vehicle_id: int, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.vehicle_id = vehicle_id
    db.commit()

    return {"message": "Vehicle assigned"}

# ---------------- CSV UPLOAD ----------------
@app.post("/upload-csv")
def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = file.file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents), encoding="utf-8", on_bad_lines="skip")
    except:
        df = pd.read_csv(io.BytesIO(contents), encoding="latin-1", on_bad_lines="skip")

    # Replace existing map orders with uploaded CSV data.
    db.query(Order).delete()
    db.commit()

    parsed_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        name = str(row.get("customer_name", "Unknown")).strip() or "Unknown"
        address = str(row.get("address", "")).strip()
        lat_raw = row.get("latitude", row.get("lat"))
        lng_raw = row.get("longitude", row.get("lng"))
        lat = None
        lng = None
        try:
            if lat_raw is not None and lng_raw is not None and str(lat_raw).strip() != "" and str(lng_raw).strip() != "":
                lat = float(lat_raw)
                lng = float(lng_raw)
        except (TypeError, ValueError):
            lat, lng = None, None

        if address or (lat is not None and lng is not None):
            parsed_rows.append({"name": name, "address": address, "lat": lat, "lng": lng})

    unique_addresses = list({item["address"] for item in parsed_rows if item["address"] and item["lat"] is None})
    geocode_cache: dict[str, tuple[float | None, float | None]] = {}

    # Geocode unique addresses in parallel to avoid upload timeouts.
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_address = {executor.submit(geocode, address): address for address in unique_addresses}
        for future in as_completed(future_to_address):
            address = future_to_address[future]
            try:
                geocode_cache[address] = future.result()
            except Exception:
                geocode_cache[address] = (None, None)

    results = []
    for item in parsed_rows:
        name = item["name"]
        address = item["address"] or "Unknown Address"
        lat = item["lat"]
        lng = item["lng"]
        if lat is None:
            lat, lng = geocode_cache.get(address, (None, None))
        if lat is None:
            continue
        order = Order(
            customer_name=name,
            address=address,
            latitude=lat,
            longitude=lng,
            status="pending",
        )
        db.add(order)
        results.append(name)

    db.commit()

    saved = len(results)
    if saved == 0:
        return {
            "success": False,
            "message": "No locations could be mapped from CSV",
            "error": "Could not geocode any address. Add valid OPENCAGE_API_KEY or include lat/lng columns in CSV.",
            "total": len(df),
            "saved": 0
        }

    return {
        "success": True,
        "message": "CSV uploaded",
        "total": len(df),
        "saved": saved
    }

# ---------------- PDF / DOCX UPLOAD ----------------
@app.post("/upload-orders")
def upload_orders(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".pdf"):
        addresses = parse_pdf(file_path)

    elif file.filename.endswith(".docx"):
        addresses = parse_docx(file_path)

    else:
        return {"error": "Unsupported file format"}

    saved = 0

    for addr in addresses:
        lat, lng = geocode(addr)

        if lat is None:
            continue

        order = Order(
            customer_name="Unknown",
            address=addr,
            latitude=lat,
            longitude=lng,
            status="pending",
        )

        db.add(order)
        saved += 1

    db.commit()

    return {
        "message": "File processed",
        "total": len(addresses),
        "saved": saved
    }

# ---------------- ROUTE ENGINE SCHEMAS ----------------
class OptimizeRequest(BaseModel):
    orders: list[dict]
    vehicle_count: int = 3

class RerouteRequest(BaseModel):
    vehicle_id: str
    current_position: dict
    remaining_orders: list[dict]
    reason: str = "disruption detected"


class VehicleGpsUpdate(BaseModel):
    vehicle_id: str
    lat: float
    lng: float
    reason: str | None = None


class DispatcherVehicleAction(BaseModel):
    vehicle_id: str
    reason: str | None = None


class DispatcherManualReroute(BaseModel):
    vehicle_id: str
    reason: str
    lat: float | None = None
    lng: float | None = None


# ---------------- ROUTE ENGINE APIs ----------------
@app.post("/optimize-routes")
def optimize_routes(body: OptimizeRequest):
    if not body.orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    if body.vehicle_count < 1:
        raise HTTPException(status_code=400, detail="vehicle_count must be >= 1")
    return run_optimization(body.orders, body.vehicle_count)


@app.get("/risk-score")
def risk_score(
    rain_mm: float = 0,
    wind_speed_ms: float = 0,
    visibility_km: float = 10,
    traffic_delay_min: float = 0,
    temperature_c: float = 25,
):
    features = [rain_mm, wind_speed_ms, visibility_km, traffic_delay_min, temperature_c]
    return compute_risk_score(features)


@app.post("/reroute")
def reroute(body: RerouteRequest):
    # Compatibility endpoint: preserve the incoming order sequence (do NOT reorder).
    if not body.remaining_orders:
        return {"success": True, "vehicle_id": body.vehicle_id, "message": "No remaining orders", "new_route": [], "reason": body.reason}

    waypoints = [body.current_position] + body.remaining_orders + [WAREHOUSE]
    route_coordinates, routing_method, routing_debug = build_route_coordinates(waypoints)
    total_km = _path_distance_km(route_coordinates)
    new_eta_minutes = int((total_km / 25.0) * 60.0) + len(body.remaining_orders) * 2
    return {
        "success": True,
        "vehicle_id": body.vehicle_id,
        "reason": body.reason,
        "new_route": {
            "assigned_orders": body.remaining_orders,
            "route_coordinates": route_coordinates,
            "total_distance_km": round(total_km, 2),
            "new_eta_minutes": new_eta_minutes,
            "routing_method": routing_method,
            "routing_debug": routing_debug,
        },
    }


@app.post("/api/vehicle_gps_update")
def api_vehicle_gps_update(body: VehicleGpsUpdate, db: Session = Depends(get_db)):
    """
    Live GPS update. Detect deviation from planned route and auto-trigger reroute.
    Preserves stop list + order sequence; only path/timing changes.
    """
    vehicle_id = body.vehicle_id
    lat = float(body.lat)
    lng = float(body.lng)
    reason = body.reason or "Route deviation detected"

    active = next((r for r in (simulation_state.get("active_routes") or []) if r.get("vehicle_id") == vehicle_id), None)
    if not active:
        raise HTTPException(status_code=404, detail="Vehicle not found or no active route")

    coords = active.get("route_coordinates") or []
    cur_idx = int(active.get("current_index", 0))
    min_m, nearest_i = _min_distance_m_to_polyline(lat, lng, coords, start_idx=cur_idx, window=160)
    active["current_index"] = int(nearest_i)

    # Deviation threshold (meters)
    threshold_m = float(os.getenv("DEVIATION_THRESHOLD_M", "150"))
    deviated = bool(min_m > threshold_m)

    if not deviated:
        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "deviated": False,
            "distance_to_route_m": round(min_m, 1),
            "nearest_index": int(nearest_i),
        }

    # Trigger reroute from CURRENT GPS location through remaining stops in same order.
    reroute_result = _reroute_preserve_sequence(
        vehicle_id=vehicle_id,
        current_position={"lat": lat, "lng": lng},
        reason=reason,
        db=db,
    )
    reroute_result["deviated"] = True
    reroute_result["distance_to_route_m"] = round(min_m, 1)
    reroute_result["threshold_m"] = threshold_m
    return reroute_result


@app.get("/api/dispatcher/state")
def api_dispatcher_state():
    return {
        "success": True,
        "active_vehicles": simulation_state.get("active_vehicles", []),
        "paused_vehicles": simulation_state.get("paused_vehicles", []),
        "dispatcher_events": simulation_state.get("dispatcher_events", [])[-50:],
    }


@app.post("/api/dispatcher/pause")
def api_dispatcher_pause(body: DispatcherVehicleAction):
    if body.vehicle_id not in (simulation_state.get("active_vehicles") or []):
        raise HTTPException(status_code=404, detail="Vehicle not active")
    _set_vehicle_paused(body.vehicle_id, True, body.reason or "Paused by dispatcher")
    return {"success": True, "vehicle_id": body.vehicle_id, "paused": True}


@app.post("/api/dispatcher/resume")
def api_dispatcher_resume(body: DispatcherVehicleAction):
    if body.vehicle_id not in (simulation_state.get("active_vehicles") or []):
        raise HTTPException(status_code=404, detail="Vehicle not active")
    _set_vehicle_paused(body.vehicle_id, False, body.reason or "Resumed by dispatcher")
    return {"success": True, "vehicle_id": body.vehicle_id, "paused": False}


@app.post("/api/dispatcher/manual-reroute")
def api_dispatcher_manual_reroute(body: DispatcherManualReroute, db: Session = Depends(get_db)):
    active = next((r for r in (simulation_state.get("active_routes") or []) if r.get("vehicle_id") == body.vehicle_id), None)
    if not active:
        raise HTTPException(status_code=404, detail="Vehicle not found or no active route")
    coords = active.get("route_coordinates") or []
    idx = int(active.get("current_index", 0))
    if not coords:
        raise HTTPException(status_code=400, detail="No route geometry for this vehicle")
    idx = max(0, min(idx, len(coords) - 1))
    if body.lat is not None and body.lng is not None:
        # Prefer live UI marker position for immediate, accurate reroute origin.
        current_position = {"lat": float(body.lat), "lng": float(body.lng)}
    else:
        current_position = {"lat": float(coords[idx][0]), "lng": float(coords[idx][1])}
    result = _reroute_preserve_sequence(
        vehicle_id=body.vehicle_id,
        current_position=current_position,
        reason=f"DISPATCHER: {body.reason}",
        db=db,
    )
    simulation_state["dispatcher_events"].append({
        "type": "manual_reroute",
        "vehicle_id": body.vehicle_id,
        "reason": body.reason,
        "timestamp": datetime.now().isoformat(),
        "updated_eta_min": result.get("updated_eta_min"),
        "updated_distance_km": result.get("updated_distance_km"),
    })
    return result


# ---------------- WEATHER API ----------------
@app.get("/weather/{order_id}")
async def weather(order_id: int, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()

    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    weather_data = await get_weather(order.latitude, order.longitude)
    score = calculate_weather_score(weather_data)

    return {
        "order_id": order.id,
        "weather_score": score,
        "weather": weather_data
    }