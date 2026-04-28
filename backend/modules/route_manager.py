import math
import uuid
from datetime import datetime
from typing import List, Dict

class Point:
    def __init__(self, lat: float, lon: float, name: str = ""):
        self.lat = lat
        self.lon = lon
        self.name = name

def haversine_distance(p1: Point, p2: Point) -> float:
    R = 6371.0 # Earth radius in km
    lat1_rad = math.radians(p1.lat)
    lat2_rad = math.radians(p2.lat)
    dlat = math.radians(p2.lat - p1.lat)
    dlon = math.radians(p2.lon - p1.lon)
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class Delivery:
    def __init__(self, delivery_id: str, original_route: List[Point], alternative_route: List[Point] = None, speed_kmh: float = 80.0):
        self.id = delivery_id
        self.original_route = original_route
        self.alternative_route = alternative_route if alternative_route else original_route
        
        self.current_route = list(self.original_route)
        self.current_waypoint_index = 0
        self.vehicle_location = Point(self.current_route[0].lat, self.current_route[0].lon, f"Vehicle {self.id} Location")
        self.speed_kmh = speed_kmh
        self.is_rerouted = False
        
        # Delivery specific state
        self.weather_info = {}
        self.weather_score = 0.0
        
        # Event History Log
        self.event_log = []
        
        # Initial Event
        self.add_event(
            "Order Dispatched", 
            f"Package dispatched from {self.current_route[0].name}. Headed to {self.current_route[-1].name}.",
            "normal"
        )

    def add_event(self, title: str, description: str, event_type: str = "normal"):
        self.event_log.insert(0, {
            "id": len(self.event_log) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "description": description,
            "type": event_type, # normal, warning, danger, success
            "current_eta": round(self.get_eta_hours(), 2)
        })

    def step_vehicle(self, time_delta_hours: float):
        if self.current_waypoint_index >= len(self.current_route) - 1:
            # Check if arrived and log
            if len(self.event_log) > 0 and self.event_log[0]["title"] != "Order Delivered":
                self.add_event("Order Delivered", f"Package has arrived at {self.current_route[-1].name}.", "success")
            return
            
        target = self.current_route[self.current_waypoint_index + 1]
        dist_to_target = haversine_distance(self.vehicle_location, target)
        dist_can_travel = self.speed_kmh * time_delta_hours
        
        if dist_can_travel >= dist_to_target:
            self.vehicle_location.lat = target.lat
            self.vehicle_location.lon = target.lon
            self.current_waypoint_index += 1
            
            # Log waypoint reached
            if self.current_waypoint_index < len(self.current_route) - 1:
                self.add_event("Checkpoint Reached", f"Vehicle reached {target.name}.", "normal")
        else:
            ratio = dist_can_travel / dist_to_target if dist_to_target > 0 else 0
            self.vehicle_location.lat += (target.lat - self.vehicle_location.lat) * ratio
            self.vehicle_location.lon += (target.lon - self.vehicle_location.lon) * ratio

    def get_eta_hours(self) -> float:
        if self.current_waypoint_index >= len(self.current_route) - 1:
            return 0.0
            
        total_dist = 0.0
        target = self.current_route[self.current_waypoint_index + 1]
        total_dist += haversine_distance(self.vehicle_location, target)
        
        for i in range(self.current_waypoint_index + 1, len(self.current_route) - 1):
            total_dist += haversine_distance(self.current_route[i], self.current_route[i+1])
            
        return total_dist / self.speed_kmh

    def trigger_reroute(self, reason: str = "Severe weather condition detected"):
        if not self.is_rerouted and self.alternative_route:
            self.is_rerouted = True
            
            old_eta = self.get_eta_hours()
            # Simplistic reroute: connect current location to the rest of the alternative route
            # Assuming alternative route and original route diverge, we just replace remaining
            self.current_route = [self.vehicle_location] + self.alternative_route[2:]
            self.current_waypoint_index = 0
            new_eta = self.get_eta_hours()
            
            self.add_event(
                "Path Change (Rerouted)",
                f"{reason}. Rerouted to alternative safer path. ETA rescheduled from {round(old_eta, 2)}h to {round(new_eta, 2)}h.",
                "danger"
            )

    def get_state(self) -> dict:
        return {
            "id": self.id,
            "origin": self.current_route[0].name,
            "destination": self.current_route[-1].name,
            "location": {"lat": self.vehicle_location.lat, "lon": self.vehicle_location.lon},
            "route": [{"lat": p.lat, "lon": p.lon, "name": p.name} for p in self.current_route],
            "eta_hours": round(self.get_eta_hours(), 2),
            "is_rerouted": self.is_rerouted,
            "weather": self.weather_info,
            "weather_score": self.weather_score,
            "event_log": self.event_log
        }

class RouteManager:
    def __init__(self):
        self.deliveries: List[Delivery] = []
        
        # Delivery 1: Mumbai to Pune
        route1 = [
            Point(19.0760, 72.8777, "Mumbai"),
            Point(19.0330, 73.0297, "Navi Mumbai"),
            Point(18.7481, 73.4072, "Lonavala"),
            Point(18.5204, 73.8567, "Pune")
        ]
        alt1 = [
            Point(19.0760, 72.8777, "Mumbai"),
            Point(18.9894, 73.1175, "Panvel"),
            Point(18.7891, 73.3444, "Khopoli"),
            Point(18.5204, 73.8567, "Pune")
        ]
        self.deliveries.append(Delivery("DEL-101", route1, alt1, 80.0))
        
        # Delivery 2: Delhi to Jaipur
        route2 = [
            Point(28.6139, 77.2090, "Delhi"),
            Point(28.4595, 77.0266, "Gurugram"),
            Point(28.1706, 76.6201, "Rewari"),
            Point(26.9124, 75.7873, "Jaipur")
        ]
        self.deliveries.append(Delivery("DEL-102", route2, route2, 85.0))
        
        # Delivery 3: Bangalore to Mysore
        route3 = [
            Point(12.9716, 77.5946, "Bangalore"),
            Point(12.7271, 77.2764, "Ramanagara"),
            Point(12.5207, 76.8967, "Mandya"),
            Point(12.2958, 76.6394, "Mysore")
        ]
        self.deliveries.append(Delivery("DEL-103", route3, route3, 70.0))

    def step_all(self, time_delta_hours: float):
        for delivery in self.deliveries:
            delivery.step_vehicle(time_delta_hours)
            
    def get_all_states(self) -> List[dict]:
        return [delivery.get_state() for delivery in self.deliveries]
