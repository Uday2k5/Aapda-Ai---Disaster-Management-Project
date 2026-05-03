from __future__ import annotations

import heapq
import json
import math
import urllib.parse
import urllib.request
from typing import Any

from backend.earthquake_service import _seismic_prior, predict_earthquake
from backend.flood_service import _clamp, _regional_flood_factor, flood_risk_for_location


def safest_path_for_location(disaster: str, latitude: float, longitude: float, depth: float = 10.0) -> dict[str, Any]:
    radius_used = 6
    if disaster == "flood":
        center = flood_risk_for_location(latitude, longitude)
        center_score = float(center["risk_score"])
        if center_score < 0.55:
            return _no_route(disaster, center)
        grid, radius_used = _expand_flood_grid(latitude, longitude, center)
    elif disaster == "earthquake":
        center = predict_earthquake(latitude, longitude, depth)
        center_score = float(center["risk_score"])
        if center_score < 0.62:
            return _no_route(disaster, center)
        grid, radius_used = _expand_earthquake_grid(latitude, longitude, depth, center)
    else:
        raise ValueError("Unsupported disaster type")

    goal = _best_exit_cell(grid, radius_used)
    if goal is None:
        return {
            "disaster": disaster,
            "needed": False,
            "message": "No safer corridor was found in the sampled area.",
            "start_risk_score": round(center_score, 3),
            "start_risk_level": center["risk_level"],
            "path": [],
        }

    route_nodes = _astar_path(grid, radius_used, goal)
    exit_cell = grid[goal]

    route_message = (
        f"Suggested path exits toward a {exit_cell['risk_level'].lower()}-risk edge zone."
        if exit_cell["risk_level"] in {"Low", "Moderate"}
        else "Suggested path follows the least-risk corridor available in the sampled region."
    )

    road_route = _road_route(latitude, longitude, exit_cell["lat"], exit_cell["lon"])
    if road_route["ok"]:
        path = road_route["geometry"]
        steps = road_route["steps"]
        route_mode = "road"
    else:
        path = [[grid[node]["lat"], grid[node]["lon"]] for node in route_nodes]
        steps = []
        route_mode = "grid-fallback"

    return {
        "disaster": disaster,
        "needed": True,
        "message": route_message,
        "start_risk_score": round(center_score, 3),
        "start_risk_level": center["risk_level"],
        "end_risk_score": round(exit_cell["risk_score"], 3),
        "end_risk_level": exit_cell["risk_level"],
        "search_radius": radius_used,
        "route_mode": route_mode,
        "steps": steps,
        "path": path,
    }


def _no_route(disaster: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "disaster": disaster,
        "needed": False,
        "message": "A shortest safe path is not needed because the current zone is below the high-risk threshold.",
        "start_risk_score": round(float(result["risk_score"]), 3),
        "start_risk_level": result["risk_level"],
        "path": [],
    }


def _expand_flood_grid(
    latitude: float, longitude: float, center: dict[str, Any]
) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    for radius in (6, 9, 12):
        grid = _build_flood_grid(latitude, longitude, center, radius)
        if _has_safe_boundary(grid, radius):
            return grid, radius
    return grid, radius


def _build_flood_grid(
    latitude: float, longitude: float, center: dict[str, Any], radius: int
) -> dict[tuple[int, int], dict[str, Any]]:
    step = 0.02
    weather = center["weather"]
    rain_now = float(weather.get("rain_now_mm", 0.0))
    rain_24h = float(weather.get("rain_24h_mm", 0.0))
    rain_72h = float(weather.get("rain_72h_mm", 0.0))
    rainfall_score = min(0.62, rain_now / 35 * 0.18 + rain_24h / 120 * 0.26 + rain_72h / 240 * 0.18)

    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for gx in range(-radius, radius + 1):
        for gy in range(-radius, radius + 1):
            lat = latitude + gy * step
            lon = longitude + gx * step
            regional_factor, _ = _regional_flood_factor(lat, lon)
            risk = _clamp(0.08 + rainfall_score + regional_factor, 0.02, 0.98)
            level = "High" if risk >= 0.55 else "Moderate" if risk >= 0.35 else "Low"
            grid[(gx, gy)] = {"lat": round(lat, 5), "lon": round(lon, 5), "risk_score": risk, "risk_level": level}
    return grid


def _expand_earthquake_grid(
    latitude: float, longitude: float, depth: float, center: dict[str, Any]
) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    for radius in (6, 9, 12):
        grid = _build_earthquake_grid(latitude, longitude, depth, center, radius)
        if _has_safe_boundary(grid, radius):
            return grid, radius
    return grid, radius


def _build_earthquake_grid(
    latitude: float, longitude: float, depth: float, center: dict[str, Any], radius: int
) -> dict[tuple[int, int], dict[str, Any]]:
    step = 0.03
    center_prior, _ = _seismic_prior(latitude, longitude)
    center_score = float(center["risk_score"])
    base_score = max(0.03, center_score - center_prior)

    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for gx in range(-radius, radius + 1):
        for gy in range(-radius, radius + 1):
            lat = latitude + gy * step
            lon = longitude + gx * step
            prior, _ = _seismic_prior(lat, lon)
            risk = min(0.98, max(0.03, base_score + prior))
            level = "High" if risk >= 0.62 else "Moderate" if risk >= 0.38 else "Low"
            grid[(gx, gy)] = {"lat": round(lat, 5), "lon": round(lon, 5), "risk_score": risk, "risk_level": level}
    return grid


def _best_exit_cell(grid: dict[tuple[int, int], dict[str, Any]], radius: int) -> tuple[int, int] | None:
    boundary: list[tuple[float, tuple[int, int]]] = []
    preferred: list[tuple[float, tuple[int, int]]] = []
    for node, cell in grid.items():
        x, y = node
        if abs(x) == radius or abs(y) == radius:
            boundary.append((cell["risk_score"], node))
            if cell["risk_level"] == "Low":
                preferred.append((cell["risk_score"], node))
    if preferred:
        return min(preferred, key=lambda item: item[0])[1]
    if boundary:
        return min(boundary, key=lambda item: item[0])[1]
    return None


def _has_safe_boundary(grid: dict[tuple[int, int], dict[str, Any]], radius: int) -> bool:
    for (x, y), cell in grid.items():
        if abs(x) == radius or abs(y) == radius:
            if cell["risk_level"] in {"Low", "Moderate"}:
                return True
    return False


def _astar_path(
    grid: dict[tuple[int, int], dict[str, Any]], radius: int, goal: tuple[int, int]
) -> list[tuple[int, int]]:
    start = (0, 0)
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in _neighbors(current, radius):
            if neighbor not in grid:
                continue
            risk = float(grid[neighbor]["risk_score"])
            move_cost = _distance(current, neighbor) * (1.0 + risk * 5.0)
            new_cost = cost_so_far[current] + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _distance(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return [start]

    node: tuple[int, int] | None = goal
    path: list[tuple[int, int]] = []
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def _neighbors(node: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    x, y = node
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if -radius <= nx <= radius and -radius <= ny <= radius:
                neighbors.append((nx, ny))
    return neighbors


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _road_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> dict[str, Any]:
    coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    params = urllib.parse.urlencode(
        {
            "overview": "full",
            "steps": "true",
            "geometries": "geojson",
        }
    )
    url = f"https://router.project-osrm.org/route/v1/driving/{coords}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("code") != "Ok" or not payload.get("routes"):
            return {"ok": False, "error": payload.get("code", "NoRoute")}

        route = payload["routes"][0]
        geometry = [[lat, lon] for lon, lat in route["geometry"]["coordinates"]]
        steps: list[dict[str, Any]] = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                maneuver = step.get("maneuver", {})
                step_type = maneuver.get("type", "continue")
                modifier = maneuver.get("modifier", "")
                road_name = step.get("name") or "unnamed road"
                prefix = f"{step_type} {modifier}".strip()
                steps.append(
                    {
                        "instruction": f"{prefix} onto {road_name}",
                        "distance_m": round(float(step.get("distance", 0.0))),
                        "duration_s": round(float(step.get("duration", 0.0))),
                    }
                )
        return {"ok": True, "geometry": geometry, "steps": steps[:8]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
