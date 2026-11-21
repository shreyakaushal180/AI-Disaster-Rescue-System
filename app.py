# AIDRA - Adaptive Intelligent Disaster Response Agent
# Web Application & Autonomous Emergency Command Center

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

from flask import Flask, jsonify, render_template_string, request, send_from_directory, session

from agent import choose_best_victim, compute_path_risk, get_severity_score
from csp import csp_assign
from environment import Environment
from fuzzy import fuzzy_inference, explain_uncertainty_impact
from ml_model import get_model_metrics, predict_priority
from search import bfs, dfs, a_star, a_star_risk, greedy_best_first, hill_climbing

PROJECT_DIR = Path(__file__).resolve().parent
SIMULATION_SCRIPT = PROJECT_DIR / "main.py"
ROUTE_RISK_WEIGHT = 3

app = Flask(__name__)
app.secret_key = os.environ.get("AIDRA_SECRET_KEY", "dev-only-change-me")

ACTIVE_SCENARIO_SESSION_KEY = "active_scenario"

DEFAULT_SCENARIO = {
    "size": 10,
    "base": [0, 0],
    "hospitals": [[9, 9], [6, 8]],
    "victims": [
        {"id": 1, "x": 2, "y": 2, "severity": "critical"},
        {"id": 2, "x": 6, "y": 2, "severity": "critical"},
        {"id": 3, "x": 2, "y": 7, "severity": "moderate"},
        {"id": 4, "x": 8, "y": 6, "severity": "minor"},
    ],
    "risk_zones": [
        {"x": 1, "y": 1, "level": 2},
        {"x": 5, "y": 3, "level": 2},
        {"x": 4, "y": 7, "level": 3},
        {"x": 7, "y": 8, "level": 2},
        {"x": 0, "y": 4, "level": 2},
        {"x": 1, "y": 4, "level": 2},
    ],
    "blocked_cells": [[4, 4], [5, 4], [3, 5], [5, 6]],
}

PRESETS = {
    "default": {
        "name": "Standard Balanced Mission",
        "badge": "Default Balanced",
        "size": 10,
        "description": "City grid with 4 victims, corridor hazard zones, and dual hospital access.",
        "scenario": DEFAULT_SCENARIO
    },
    "flood": {
        "name": "Urban Flood & Bridge Collapse",
        "badge": "Water Barrier & Cutoffs",
        "size": 10,
        "description": "Flash flooding cuts off main transit arteries. Demonstrates obstacle rerouting.",
        "scenario": {
            "size": 10,
            "base": [0, 0],
            "hospitals": [[9, 9], [1, 8]],
            "victims": [
                {"id": 1, "x": 3, "y": 3, "severity": "critical"},
                {"id": 2, "x": 7, "y": 3, "severity": "critical"},
                {"id": 3, "x": 8, "y": 7, "severity": "moderate"},
                {"id": 4, "x": 2, "y": 8, "severity": "minor"},
            ],
            "risk_zones": [
                {"x": 2, "y": 4, "level": 2},
                {"x": 3, "y": 4, "level": 2},
                {"x": 4, "y": 4, "level": 3},
                {"x": 5, "y": 4, "level": 3},
                {"x": 6, "y": 4, "level": 2},
            ],
            "blocked_cells": [[4, 1], [4, 2], [4, 5], [4, 6], [5, 5]],
        }
    },
    "wildfire": {
        "name": "Wildfire Inferno Incident",
        "badge": "Extreme Thermal Firestorm",
        "size": 10,
        "description": "Thermal firestorm across central corridor. Demonstrates Risk-Aware A* detour.",
        "scenario": {
            "size": 10,
            "base": [0, 0],
            "hospitals": [[9, 9], [9, 2]],
            "victims": [
                {"id": 1, "x": 4, "y": 2, "severity": "critical"},
                {"id": 2, "x": 8, "y": 3, "severity": "critical"},
                {"id": 3, "x": 3, "y": 6, "severity": "moderate"},
                {"id": 4, "x": 7, "y": 7, "severity": "minor"},
            ],
            "risk_zones": [
                {"x": 1, "y": 2, "level": 3},
                {"x": 2, "y": 2, "level": 3},
                {"x": 3, "y": 2, "level": 3},
                {"x": 2, "y": 3, "level": 2},
                {"x": 5, "y": 5, "level": 3},
                {"x": 6, "y": 5, "level": 3},
                {"x": 7, "y": 5, "level": 2},
            ],
            "blocked_cells": [[5, 2], [5, 3]],
        }
    },
    "mass_casualty": {
        "name": "Mass Casualty Incident",
        "badge": "CSP Resource Stress Test",
        "size": 10,
        "description": "6 victims with 2 ambulances. Demonstrates triage and repeated hospital dispatches.",
        "scenario": {
            "size": 10,
            "base": [0, 0],
            "hospitals": [[9, 9], [7, 8]],
            "victims": [
                {"id": 1, "x": 2, "y": 2, "severity": "critical"},
                {"id": 2, "x": 5, "y": 2, "severity": "critical"},
                {"id": 3, "x": 2, "y": 6, "severity": "moderate"},
                {"id": 4, "x": 7, "y": 5, "severity": "moderate"},
                {"id": 5, "x": 4, "y": 8, "severity": "minor"},
                {"id": 6, "x": 8, "y": 7, "severity": "minor"},
            ],
            "risk_zones": [
                {"x": 3, "y": 4, "level": 2},
                {"x": 6, "y": 3, "level": 2},
            ],
            "blocked_cells": [[3, 3], [6, 4]],
        }
    },
    "earthquake": {
        "name": "Major Earthquake (Rubble Maze)",
        "badge": "Collapsed City Alleys",
        "size": 12,
        "description": "Massive building collapse creates narrow labyrinth corridors. Tests maze navigation.",
        "scenario": {
            "size": 12,
            "base": [0, 0],
            "hospitals": [[11, 11], [1, 10]],
            "victims": [
                {"id": 1, "x": 4, "y": 3, "severity": "critical"},
                {"id": 2, "x": 9, "y": 4, "severity": "critical"},
                {"id": 3, "x": 3, "y": 8, "severity": "moderate"},
                {"id": 4, "x": 8, "y": 9, "severity": "minor"},
                {"id": 5, "x": 10, "y": 2, "severity": "moderate"},
            ],
            "risk_zones": [
                {"x": 2, "y": 2, "level": 2},
                {"x": 5, "y": 5, "level": 3},
                {"x": 7, "y": 7, "level": 2},
                {"x": 6, "y": 2, "level": 2},
            ],
            "blocked_cells": [
                [2, 3], [3, 3], [5, 3], [6, 3],
                [3, 5], [4, 5], [6, 5], [7, 5],
                [5, 7], [6, 7], [8, 7], [9, 7]
            ],
        }
    },
    "hazmat": {
        "name": "Industrial Toxic Hazmat Leak",
        "badge": "Chemical Zone Perimeter",
        "size": 10,
        "description": "Chemical spill expands outward with concentric risk rings. Tests safety detouring.",
        "scenario": {
            "size": 10,
            "base": [0, 0],
            "hospitals": [[9, 9], [0, 9]],
            "victims": [
                {"id": 1, "x": 3, "y": 2, "severity": "critical"},
                {"id": 2, "x": 7, "y": 3, "severity": "critical"},
                {"id": 3, "x": 2, "y": 7, "severity": "moderate"},
                {"id": 4, "x": 8, "y": 7, "severity": "minor"},
            ],
            "risk_zones": [
                {"x": 4, "y": 4, "level": 3}, {"x": 4, "y": 5, "level": 3},
                {"x": 5, "y": 4, "level": 3}, {"x": 5, "y": 5, "level": 3},
                {"x": 3, "y": 4, "level": 2}, {"x": 3, "y": 5, "level": 2},
                {"x": 6, "y": 4, "level": 2}, {"x": 6, "y": 5, "level": 2},
                {"x": 4, "y": 3, "level": 2}, {"x": 5, "y": 3, "level": 2},
                {"x": 4, "y": 6, "level": 2}, {"x": 5, "y": 6, "level": 2},
            ],
            "blocked_cells": [[3, 3], [6, 6]],
        }
    },
    "dense_urban": {
        "name": "Downtown Megacity Multi-Hospital",
        "badge": "12x12 & 3 Safe Hospitals",
        "size": 12,
        "description": "Large urban sector with 3 destination hospitals and 6 casualties.",
        "scenario": {
            "size": 12,
            "base": [0, 0],
            "hospitals": [[11, 11], [11, 0], [0, 11]],
            "victims": [
                {"id": 1, "x": 2, "y": 3, "severity": "critical"},
                {"id": 2, "x": 6, "y": 3, "severity": "critical"},
                {"id": 3, "x": 4, "y": 8, "severity": "moderate"},
                {"id": 4, "x": 9, "y": 7, "severity": "minor"},
                {"id": 5, "x": 8, "y": 2, "severity": "moderate"},
                {"id": 6, "x": 10, "y": 5, "severity": "critical"},
            ],
            "risk_zones": [
                {"x": 3, "y": 3, "level": 2}, {"x": 5, "y": 3, "level": 2},
                {"x": 4, "y": 6, "level": 3}, {"x": 8, "y": 5, "level": 2},
            ],
            "blocked_cells": [[4, 4], [5, 4], [4, 5], [5, 5], [7, 7], [8, 7]],
        }
    },
    "island": {
        "name": "Islands & Bottleneck Bridges",
        "badge": "Single Bridge Choke Point",
        "size": 10,
        "description": "Water divide where only a single bridge coordinate (3, 4) allows access across.",
        "scenario": {
            "size": 10,
            "base": [0, 0],
            "hospitals": [[9, 9]],
            "victims": [
                {"id": 1, "x": 2, "y": 2, "severity": "critical"},
                {"id": 2, "x": 7, "y": 2, "severity": "critical"},
                {"id": 3, "x": 3, "y": 7, "severity": "moderate"},
                {"id": 4, "x": 8, "y": 6, "severity": "minor"},
            ],
            "risk_zones": [
                {"x": 1, "y": 1, "level": 2},
                {"x": 6, "y": 3, "level": 2},
                {"x": 4, "y": 7, "level": 3},
            ],
            "blocked_cells": [
                [0, 4], [1, 4], [2, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4]
            ],
        }
    }
}

MAX_FORM_VICTIMS = 16
MAX_FORM_RISK_ZONES = 24
MAX_FORM_BLOCKED_CELLS = 36


def quiet_call(func, *args, **kwargs):
    with redirect_stdout(StringIO()):
        return func(*args, **kwargs)


def get_active_scenario():
    return session.get(ACTIVE_SCENARIO_SESSION_KEY, DEFAULT_SCENARIO)


def set_active_scenario(scenario):
    session[ACTIVE_SCENARIO_SESSION_KEY] = normalize_scenario(scenario)


def clear_active_scenario():
    session.pop(ACTIVE_SCENARIO_SESSION_KEY, None)


def parse_int(value, field):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.")


def parse_optional_position(form, x_key, y_key, field):
    x_value = form.get(x_key, "").strip()
    y_value = form.get(y_key, "").strip()
    if not x_value and not y_value:
        return None
    if not x_value or not y_value:
        raise ValueError(f"{field} needs both x and y values.")
    return [parse_int(x_value, f"{field} x"), parse_int(y_value, f"{field} y")]


def parse_position(value, field, size):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be [x, y].")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"{field} position [{x}, {y}] is outside the {size}x{size} grid.")
    return x, y


def normalize_scenario(scenario=None):
    scenario = scenario or DEFAULT_SCENARIO
    if not isinstance(scenario, dict):
        raise ValueError("Scenario must be a JSON object.")

    size = scenario.get("size", 10)
    if not isinstance(size, int) or size < 4 or size > 20:
        raise ValueError("size must be an integer between 4 and 20.")

    base = parse_position(scenario.get("base", [0, 0]), "base", size)
    hospitals = [parse_position(item, "hospital", size) for item in scenario.get("hospitals", [])]
    if not hospitals:
        raise ValueError("At least one hospital is required.")

    victims = []
    for index, item in enumerate(scenario.get("victims", []), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"victims[{index}] must be an object.")
        severity = item.get("severity", "minor")
        if severity not in {"critical", "moderate", "minor"}:
            raise ValueError(f"victims[{index}].severity must be critical, moderate, or minor.")
        x, y = parse_position([item.get("x"), item.get("y")], f"victims[{index}]", size)
        victims.append({"id": int(item.get("id", index)), "x": x, "y": y, "severity": severity})

    if not victims:
        raise ValueError("At least one casualty (victim) is required.")

    victim_cells = {(item["x"], item["y"]): item["id"] for item in victims}
    if base in victim_cells:
        raise ValueError(f"Base overlaps casualty V{victim_cells[base]} at ({base[0]}, {base[1]}).")
    for index, (hx, hy) in enumerate(hospitals, start=1):
        if (hx, hy) in victim_cells:
            raise ValueError(f"Hospital {index} overlaps casualty V{victim_cells[(hx, hy)]} at ({hx}, {hy}).")

    risk_zones = []
    seen_risks = set()
    for index, item in enumerate(scenario.get("risk_zones", []), start=1):
        if not isinstance(item, dict):
            continue
        x, y = parse_position([item.get("x"), item.get("y")], f"risk_zones[{index}]", size)
        if (x, y) in victim_cells or (x, y) == base or (x, y) in hospitals:
            continue
        if (x, y) in seen_risks:
            continue
        level = int(item.get("level", 1))
        if level < 0 or level > 3:
            level = 1
        seen_risks.add((x, y))
        risk_zones.append({"x": x, "y": y, "level": level})

    blocked_cells = []
    seen_blocks = set()
    for index, item in enumerate(scenario.get("blocked_cells", []), start=1):
        try:
            x, y = parse_position(item, f"blocked_cells[{index}]", size)
            if (x, y) in victim_cells or (x, y) == base or (x, y) in hospitals or (x, y) in seen_risks:
                continue
            if (x, y) not in seen_blocks:
                seen_blocks.add((x, y))
                blocked_cells.append([x, y])
        except Exception:
            continue

    return {
        "size": size,
        "base": list(base),
        "hospitals": [list(item) for item in hospitals],
        "victims": victims,
        "risk_zones": risk_zones,
        "blocked_cells": blocked_cells,
    }


def generate_random_scenario(size=10, num_victims=4, num_hospitals=2, hazard_level="medium", block_level="medium"):
    for _ in range(50):
        base = [0, 0]
        possible_hospitals = [[size-1, size-1], [size-1, 0], [0, size-1], [size-1, size//2]]
        hospitals = possible_hospitals[:min(num_hospitals, len(possible_hospitals))]
        occupied = {tuple(base)} | {tuple(h) for h in hospitals}

        victims = []
        severities = ["critical", "critical", "moderate", "minor", "moderate", "critical"]
        for i in range(1, num_victims + 1):
            for _ in range(60):
                vx = random.randint(1, size - 2)
                vy = random.randint(1, size - 2)
                if (vx, vy) not in occupied:
                    occupied.add((vx, vy))
                    victims.append({
                        "id": i,
                        "x": vx,
                        "y": vy,
                        "severity": severities[(i - 1) % len(severities)]
                    })
                    break

        num_risks = {"low": size // 2, "medium": size + 2, "high": size * 2}.get(hazard_level, size + 2)
        risk_zones = []
        for _ in range(num_risks):
            rx = random.randint(0, size - 1)
            ry = random.randint(0, size - 1)
            if (rx, ry) not in occupied:
                lvl = random.choice([1, 2, 3])
                risk_zones.append({"x": rx, "y": ry, "level": lvl})
                occupied.add((rx, ry))

        num_blocks = {"sparse": max(2, size // 2), "medium": size + 2, "heavy": size * 2}.get(block_level, size + 2)
        blocked_cells = []
        for _ in range(num_blocks):
            bx = random.randint(0, size - 1)
            by = random.randint(0, size - 1)
            if (bx, by) not in occupied:
                blocked_cells.append([bx, by])
                occupied.add((bx, by))

        env = Environment(size)
        env.set_base(*base)
        for h in hospitals: env.add_hospital(*h)
        for v in victims: env.add_victim(v["id"], v["x"], v["y"], v["severity"])
        for r in risk_zones: env.add_risk_zone(r["x"], r["y"], r["level"])
        for b in blocked_cells: env.block_cell(*b)

        reachable = 0
        for v in victims:
            p = quiet_call(a_star, env, tuple(base), (v["x"], v["y"]))
            if p:
                for h in hospitals:
                    hp = quiet_call(a_star, env, (v["x"], v["y"]), tuple(h))
                    if hp:
                        reachable += 1
                        break
        if reachable >= 1:
            return {
                "size": size,
                "base": base,
                "hospitals": hospitals,
                "victims": victims,
                "risk_zones": risk_zones,
                "blocked_cells": blocked_cells
            }
    return DEFAULT_SCENARIO


def scenario_from_form(form):
    size = parse_int(form.get("size", 10), "Grid size")
    base = [parse_int(form.get("base_x", 0), "Base x"), parse_int(form.get("base_y", 0), "Base y")]

    hospitals = []
    for index in range(1, 7):
        pos = parse_optional_position(form, f"hospital_x_{index}", f"hospital_y_{index}", f"Hospital {index}")
        if pos:
            hospitals.append(pos)

    victims = []
    for index in range(1, MAX_FORM_VICTIMS + 1):
        pos = parse_optional_position(form, f"victim_x_{index}", f"victim_y_{index}", f"Victim {index}")
        if pos:
            victims.append({
                "id": parse_int(form.get(f"victim_id_{index}", index), f"Victim {index} id"),
                "x": pos[0],
                "y": pos[1],
                "severity": form.get(f"victim_severity_{index}", "minor"),
            })

    risk_zones = []
    for index in range(1, MAX_FORM_RISK_ZONES + 1):
        pos = parse_optional_position(form, f"risk_x_{index}", f"risk_y_{index}", f"Risk zone {index}")
        if pos:
            risk_zones.append({
                "x": pos[0],
                "y": pos[1],
                "level": parse_int(form.get(f"risk_level_{index}", 1), f"Risk zone {index} level"),
            })

    blocked_cells = []
    for index in range(1, MAX_FORM_BLOCKED_CELLS + 1):
        pos = parse_optional_position(form, f"blocked_x_{index}", f"blocked_y_{index}", f"Blocked cell {index}")
        if pos:
            blocked_cells.append(pos)

    return normalize_scenario({
        "size": size,
        "base": base,
        "hospitals": hospitals,
        "victims": victims,
        "risk_zones": risk_zones,
        "blocked_cells": blocked_cells,
    })


def pad_rows(rows, total, empty_factory):
    items = list(rows)
    for _ in range(total - len(items)):
        items.append(empty_factory())
    return items


def scenario_form_data(scenario):
    hospitals = [{"x": x, "y": y} for x, y in scenario["hospitals"]]
    victims = [{"id": v["id"], "x": v["x"], "y": v["y"], "severity": v["severity"]} for v in scenario["victims"]]
    risk_zones = [{"x": r["x"], "y": r["y"], "level": r["level"]} for r in scenario["risk_zones"]]
    blocked_cells = [{"x": x, "y": y} for x, y in scenario["blocked_cells"]]

    return {
        "size": scenario["size"],
        "base_x": scenario["base"][0],
        "base_y": scenario["base"][1],
        "hospitals": pad_rows(hospitals, 6, lambda: {"x": "", "y": ""}),
        "victims": pad_rows(victims, MAX_FORM_VICTIMS, lambda: {"id": "", "x": "", "y": "", "severity": "minor"}),
        "risk_zones": pad_rows(risk_zones, MAX_FORM_RISK_ZONES, lambda: {"x": "", "y": "", "level": 1}),
        "blocked_cells": pad_rows(blocked_cells, MAX_FORM_BLOCKED_CELLS, lambda: {"x": "", "y": ""}),
    }


def create_environment(scenario=None):
    scenario = normalize_scenario(scenario)
    env = Environment(scenario["size"])
    env.set_base(*scenario["base"])

    for hospital in scenario["hospitals"]:
        env.add_hospital(*hospital)

    for victim in scenario["victims"]:
        env.add_victim(victim["id"], victim["x"], victim["y"], victim["severity"])

    for risk_zone in scenario["risk_zones"]:
        env.add_risk_zone(risk_zone["x"], risk_zone["y"], risk_zone["level"])

    for blocked_cell in scenario["blocked_cells"]:
        env.block_cell(*blocked_cell)

    return env


def serialize_grid(env, rescue_path=None):
    rescue_path = set(rescue_path or [])
    cells = []
    hospital_labels = {
        tuple(position): f"H{index}"
        for index, position in enumerate(env.hospitals, start=1)
    }

    for x in range(env.size):
        row = []
        for y in range(env.size):
            cell = env.grid[x][y]
            terrain_type = "empty"
            terrain_label = ""
            icon = ""
            marker_label = ""
            victim_severity = ""

            if (x, y) == env.base:
                terrain_type = "base"
                terrain_label = "Base"
                marker_label = "BASE"
                icon = "🚑"
            elif cell.is_hospital:
                terrain_type = "hospital"
                marker_label = hospital_labels[(x, y)]
                terrain_label = f"Hospital {marker_label}"
                icon = "🏥"
            elif cell.is_blocked:
                terrain_type = "blocked"
                terrain_label = "Blocked road"
                marker_label = "BLOCK"
                icon = "🚧"
            elif cell.has_victim:
                terrain_type = "victim"
                victim_severity = cell.victim.severity if cell.victim else "critical"
                victim_id = cell.victim.id if cell.victim else "?"
                marker_label = f"V{victim_id}"
                terrain_label = f"Victim {marker_label} ({victim_severity})"
                icon = "🚨" if victim_severity == "critical" else ("⚠️" if victim_severity == "moderate" else "🟢")
            elif cell.risk_level > 0:
                terrain_type = f"risk risk-{cell.risk_level}"
                terrain_label = f"Fire hazard Level {cell.risk_level}"
                marker_label = f"L{cell.risk_level}"
                icon = "🔥"

            on_path = (x, y) in rescue_path

            row.append({
                "x": x,
                "y": y,
                "type": "path" if on_path else terrain_type,
                "label": "*" if on_path else marker_label,
                "terrain_type": terrain_type,
                "terrain_label": terrain_label,
                "marker_label": marker_label,
                "victim_severity": victim_severity,
                "icon": icon,
                "on_path": on_path,
                "risk": cell.risk_level,
            })
        cells.append(row)

    return cells


def find_open_cell(env):
    for x in range(env.size):
        for y in range(env.size):
            cell = env.grid[x][y]
            if not cell.is_blocked and not cell.has_victim and not cell.is_hospital and (x, y) != env.base and cell.risk_level == 0:
                return x, y
    return None


def parse_ambulance_count(value=2):
    count = parse_int(value, "Ambulance count")
    if not 1 <= count <= 6:
        raise ValueError("Ambulance count must be between 1 and 6.")
    return count


def requested_ambulance_count():
    body = request.get_json(silent=True)
    if isinstance(body, dict) and "ambulance_count" in body:
        return parse_ambulance_count(body["ambulance_count"])
    return parse_ambulance_count(
        request.args.get("ambulance_count", request.form.get("ambulance_count", 2))
    )


def _nearest_risk_aware_hospital(env, start):
    best_hospital = None
    best_path = None
    best_key = (float("inf"), float("inf"), (float("inf"), float("inf")))
    for hospital in env.hospitals:
        path = quiet_call(a_star_risk, env, start, hospital, risk_weight=ROUTE_RISK_WEIGHT)
        if path is None:
            continue
        cost = len(path) - 1 + ROUTE_RISK_WEIGHT * compute_path_risk(env, path)
        candidate_key = (cost, len(path), hospital)
        if candidate_key < best_key:
            best_key = candidate_key
            best_hospital = hospital
            best_path = path
    return best_hospital, best_path


def _plan_rescue_dispatches(env, ambulance_count):
    ambulances = [
        {"name": f"Ambulance {index}", "position": env.base, "rescued": 0}
        for index in range(1, ambulance_count + 1)
    ]
    dispatches = []
    pending = [victim for victim in env.victims if not victim.rescued]

    while pending:
        progress = False
        for ambulance in ambulances:
            candidates = []
            for victim in pending:
                goal = (victim.x, victim.y)
                outbound = quiet_call(
                    a_star_risk, env, ambulance["position"], goal,
                    risk_weight=ROUTE_RISK_WEIGHT,
                )
                if outbound is None:
                    continue
                hospital, inbound = _nearest_risk_aware_hospital(env, goal)
                if hospital is None or inbound is None:
                    continue
                route = outbound + inbound[1:]
                route_risk = compute_path_risk(env, route)
                travel_steps = len(route) - 1
                priority = get_severity_score(victim.severity) - travel_steps - route_risk
                hospital_label = f"H{env.hospitals.index(hospital) + 1}"
                candidates.append((-priority, travel_steps, victim.id, victim, hospital, hospital_label, outbound, inbound, route))

            if not candidates:
                continue

            _, travel_steps, _, victim, hospital, hospital_label, outbound, inbound, route = min(candidates, key=lambda item: item[:3])
            victim.rescued = True
            ambulance["position"] = hospital
            ambulance["rescued"] += 1
            pending.remove(victim)
            progress = True
            dispatches.append({
                "ambulance": ambulance["name"],
                "victim": victim.id,
                "severity": victim.severity,
                "location": [victim.x, victim.y],
                "from": list(route[0]),
                "hospital": list(hospital),
                "hospital_label": hospital_label,
                "path": route,
                "outbound_path": outbound,
                "hospital_path": inbound,
                "distance": travel_steps,
                "risk": route_risk,
            })

        if not progress:
            break

    return dispatches, ambulances


def build_dashboard_data(scenario=None, event=None, preset=None, ambulance_count=2):
    if preset and preset in PRESETS:
        scenario = PRESETS[preset]["scenario"]
    elif scenario is None:
        scenario = DEFAULT_SCENARIO

    scenario = normalize_scenario(scenario)

    dynamic_event_record = None
    ambulance_count = parse_ambulance_count(ambulance_count)
    ambulances_fleet = [f"Ambulance {index}" for index in range(1, ambulance_count + 1)]

    pre_env = create_environment(scenario)
    pre_victim, pre_path = quiet_call(choose_best_victim, pre_env, pre_env.base)

    victim_coords = {(v["x"], v["y"]) for v in scenario["victims"]}
    hospital_coords = {tuple(h) for h in scenario["hospitals"]}
    base_coord = tuple(scenario["base"])

    if event == "blockage":
        block_pos = (1, 2)
        if pre_path and len(pre_path) >= 3:
            for step in pre_path[1:-1]:
                if step not in victim_coords and step not in hospital_coords and step != base_coord:
                    block_pos = step
                    break
        if list(block_pos) not in scenario["blocked_cells"]:
            scenario["blocked_cells"].append(list(block_pos))
        scenario = normalize_scenario(scenario)
        dynamic_event_record = {
            "type": "blockage",
            "title": "🚧 Dynamic Replan: Road Blockage Triggered",
            "badge": "Obstacle Avoidance Replan",
            "cell": list(block_pos),
            "description": f"Road at {block_pos} was abruptly blocked by landslide rubble. Previous shortest path invalidated.",
            "ai_action": "A* search recomputed an alternate open corridor, completely avoiding the blocked cell.",
            "takeaway": "Autonomous replanning prevents ambulance trap scenarios and eliminates response delays."
        }

    elif event == "risk":
        risk_pos = (1, 3)
        if pre_path and len(pre_path) >= 2:
            for step in pre_path[1:]:
                if step not in victim_coords and step not in hospital_coords and step != base_coord:
                    risk_pos = step
                    break
        scenario["risk_zones"] = [r for r in scenario["risk_zones"] if (r["x"], r["y"]) != (risk_pos[0], risk_pos[1])]
        scenario["risk_zones"].append({"x": risk_pos[0], "y": risk_pos[1], "level": 3})
        scenario = normalize_scenario(scenario)
        dynamic_event_record = {
            "type": "risk",
            "title": "🔥 Dynamic Replan: Extreme Hazard Spike (Firestorm)",
            "badge": "Risk-Aware Rerouting",
            "cell": list(risk_pos),
            "description": f"Hazard level surged to Maximum Severity (Level 3) at {risk_pos}. Direct transit is hazardous.",
            "ai_action": "Agent automatically engaged Risk-Aware A* with hazard penalty weight=3, skirting around the fire zone.",
            "takeaway": "Recalculates the route using the updated hazard costs; residual risk is still shown in the mission metrics."
        }

    elif event == "new_victim":
        open_pos = find_open_cell(pre_env) or (5, 5)
        new_id = max([v["id"] for v in scenario["victims"]] + [0]) + 1
        scenario["victims"].append({"id": new_id, "x": open_pos[0], "y": open_pos[1], "severity": "critical"})
        scenario = normalize_scenario(scenario)
        dynamic_event_record = {
            "type": "new_victim",
            "title": "🚨 Dynamic Replan: Emergency SOS - New Critical Victim Detected",
            "badge": "Triage Priority Re-evaluation",
            "cell": list(open_pos),
            "description": f"Emergency call received for new Critical Victim V{new_id} discovered at {open_pos}.",
            "ai_action": "Agent immediately paused current dispatch queue and recalculated Priority Scores across all active victims.",
            "takeaway": "Dynamically rebalances urgency and proximity so highest-risk casualties are prioritized."
        }

    elif event == "ambulance_loss":
        ambulances_fleet = ["Ambulance 1"]
        ambulance_count = 1
        dynamic_event_record = {
            "type": "ambulance_loss",
            "title": "🚑 Dynamic Replan: Resource Depletion - Ambulance 2 Out of Service",
            "badge": "CSP Constraint Backtracking",
            "cell": None,
            "description": "Ambulance 2 is out of service. The remaining ambulance continues making hospital return trips.",
            "ai_action": "The planner recalculated assignments for the active fleet and kept dispatching until no reachable victims remained.",
            "takeaway": "The rescue queue adapts to a smaller fleet without stopping after one delivery."
        }

    env = create_environment(scenario)
    start = env.base
    ml_metrics = get_model_metrics()
    ambulances_fleet = [f"Ambulance {index}" for index in range(1, ambulance_count + 1)]
    dispatches, ambulance_states = _plan_rescue_dispatches(env, ambulance_count)
    if not dispatches:
        raise ValueError("No victim can be reached and delivered to a hospital with the current roads.")

    primary_dispatch = dispatches[0]
    victim = next(item for item in env.victims if item.id == primary_dispatch["victim"])
    selected_ids = {item["victim"] for item in dispatches}
    sev_ml = victim.severity
    base_to_victim_path = primary_dispatch["outbound_path"]
    hospital_path = primary_dispatch["hospital_path"]
    hospital = tuple(primary_dispatch["hospital"])
    full_path = primary_dispatch["path"]
    goal = (victim.x, victim.y)
    distance = len(base_to_victim_path) - 1
    risk = compute_path_risk(env, base_to_victim_path)
    knn_pred, nb_pred, dt_pred = predict_priority(distance, risk, sev_ml)
    labels = ["LOW", "MEDIUM", "HIGH"]
    ml_votes = [int(knn_pred), int(nb_pred), int(dt_pred)]
    final_ml = max(set(ml_votes), key=ml_votes.count)
    blockage_probability = 0.6
    fuzzy_result = quiet_call(fuzzy_inference, distance, risk, sev_ml, blockage_probability)
    algorithm_paths = {
        "Risk-Aware A*": a_star_risk(env, start, goal, risk_weight=ROUTE_RISK_WEIGHT),
        "A*": a_star(env, start, goal),
        "BFS": bfs(env, start, goal),
        "DFS": dfs(env, start, goal),
        "Greedy": greedy_best_first(env, start, goal),
        "Hill Climbing": hill_climbing(env, start, goal),
    }

    algorithm_profiles = {
        "Risk-Aware A*": {
            "type": "Risk-Weighted Optimal", "speed": "Fast & Heuristic", "optimality": "Optimal for weighted cost",
            "safety": "Hazard-Penalized", "badge": "Recommended for hazards",
            "explanation": f"Finds the lowest combined travel and hazard cost using f(n) = g(n) + h(n) + risk*{ROUTE_RISK_WEIGHT}. It prefers safer routes, but cannot guarantee zero risk.",
            "use_case": "Hazardous disaster zones with fire/water/gas"
        },
        "A*": {
            "type": "Optimal", "speed": "Fast & Balanced", "optimality": "Mathematically Optimal",
            "safety": "Ignores Hazard Penalty", "badge": "⚡ Shortest Path",
            "explanation": "Evaluates f(n) = g(n) + h(n) and finds a shortest open route using Manhattan distance. It does not penalize hazard cells.",
            "use_case": "General navigation with clear roads"
        },
        "BFS": {
            "type": "Exhaustive", "speed": "Slow (Explores all levels)", "optimality": "Optimal Steps",
            "safety": "Ignores Hazard Penalty", "badge": "🌐 Uniform Search",
            "explanation": "Expands neighbors uniformly in a FIFO queue. Finds shortest step count but evaluates huge state spaces without heuristic guidance.",
            "use_case": "Unweighted small graphs without hazards"
        },
        "DFS": {
            "type": "Non-Optimal", "speed": "Fast Dive (LIFO)", "optimality": "Highly Non-Optimal",
            "safety": "Extreme Risk Incurred", "badge": "❌ Non-Optimal (Long Zigzags)",
            "explanation": "Dives deep along one branch. Often produces wild 20+ step zigzags and encounters extreme hazard. Never use for emergency transport.",
            "use_case": "Maze exploration, game trees"
        },
        "Greedy": {
            "type": "Heuristic Only", "speed": "Ultra-Fast", "optimality": "Approximate",
            "safety": "Ignores Path Cost", "badge": "🎯 Greedy Best-First",
            "explanation": "Evaluates only h(n) distance to goal. Extremely quick but can easily get trapped or make sub-optimal local detours near walls.",
            "use_case": "Time-critical approximate navigation"
        },
        "Hill Climbing": {
            "type": "Local Search", "speed": "Fast", "optimality": "Prone to Local Optima",
            "safety": "High Risk of Failure", "badge": "⛰️ Local Search",
            "explanation": "Greedily moves to immediate neighbor with strictly lower heuristic. Can get completely stuck when confronting concave obstacles.",
            "use_case": "Optimization with convex landscapes"
        },
    }

    algorithm_comparison = []
    analytics_rows = []
    path_visualizations = []
    for name, algo_path in algorithm_paths.items():
        profile = algorithm_profiles.get(name, {})
        risk_score = compute_path_risk(env, algo_path) if algo_path else 0
        path_len = len(algo_path) - 1 if algo_path else 0
        reached = bool(algo_path and algo_path[-1] == goal)
        weighted_cost = path_len + ROUTE_RISK_WEIGHT * risk_score if reached else None
        algorithm_comparison.append({
            "name": name,
            "length": path_len,
            "risk": risk_score,
            "weighted_cost": weighted_cost,
            "reached": reached,
            "type": profile.get("type", "-"),
            "speed": profile.get("speed", "-"),
            "optimality": profile.get("optimality", "-"),
            "safety": profile.get("safety", "-"),
            "badge": profile.get("badge", ""),
            "explanation": profile.get("explanation", ""),
            "path": algo_path or [],
        })
        analytics_rows.append({
            "name": name,
            "length": path_len,
            "risk": risk_score,
            "weighted_cost": weighted_cost,
            "reached": reached,
            "speed": profile.get("speed", "-"),
            "optimality": profile.get("optimality", "-"),
            "safety": profile.get("safety", "-"),
            "use_case": profile.get("use_case", "-"),
        })
        preview = " -> ".join([f"({x},{y})" for x, y in (algo_path or [])[:8]])
        if algo_path and len(algo_path) > 8:
            preview = f"{preview} -> ... ({algo_path[-1][0]},{algo_path[-1][1]})"
        if not preview:
            preview = "No reachable path found"
        path_visualizations.append({"name": name, "preview": preview})

    victim_scoring = []
    for v in env.victims:
        v_goal = (v.x, v.y)
        v_path = quiet_call(a_star_risk, env, start, v_goal)
        _, v_hospital_path = _nearest_risk_aware_hospital(env, v_goal) if v_path else (None, None)
        v_route = v_path + v_hospital_path[1:] if v_path and v_hospital_path else None
        v_dist = len(v_route) - 1 if v_route else None
        v_risk = compute_path_risk(env, v_route) if v_route else None
        v_sev_score = get_severity_score(v.severity)
        if v_dist is not None and v_risk is not None:
            v_score = v_sev_score - (v_dist + v_risk)
            formula_str = f"{v_sev_score} - ({v_dist} + {v_risk}) = {v_score}"
        else:
            v_score = -999
            formula_str = "Unreachable"

        victim_scoring.append({
            "id": v.id,
            "x": v.x,
            "y": v.y,
            "severity": v.severity,
            "severity_score": v_sev_score,
            "distance": v_dist,
            "risk": v_risk,
            "score": v_score,
            "formula": formula_str,
            "selected": v.id in selected_ids,
            "rescued": v.rescued,
            "explanation": f"Severity {v_sev_score} minus round-trip travel steps ({v_dist}) and route risk ({v_risk}) gives {v_score}."
        })

    victim_scoring.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(victim_scoring, 1):
        item["rank"] = rank

    mission_grid = serialize_grid(env)
    mission_victims = [
        {
            "id": item.id,
            "x": item.x,
            "y": item.y,
            "severity": item.severity,
            "rescued": item.rescued,
            "selected": item.id in selected_ids,
        }
        for item in env.victims
    ]

    used_hospitals = {tuple(item["hospital"]) for item in dispatches}
    mission_hospitals = [
        {
            "id": index,
            "label": f"H{index}",
            "x": x,
            "y": y,
            "selected": (x, y) in used_hospitals,
        }
        for index, (x, y) in enumerate(env.hospitals, start=1)
    ]

    csp_trace_buffer = StringIO()
    with redirect_stdout(csp_trace_buffer):
        _, backtrack_count = csp_assign(env.victims, ambulances=ambulances_fleet)
    csp_trace_text = csp_trace_buffer.getvalue().strip()
    csp_trace_lines = [line for line in csp_trace_text.splitlines() if line.strip()]

    assignment = {item["victim"]: item["ambulance"] for item in dispatches}
    assigned_ids = set(assignment)
    assignment_rows = []
    for item in mission_victims:
        amb = assignment.get(item["id"])
        assignment_rows.append({
            "victim": item["id"],
            "severity": item["severity"],
            "location": f'({item["x"]}, {item["y"]})',
            "ambulance": amb if amb is not None else "Unassigned (Queued)",
            "assigned": amb is not None,
            "rescued": item["rescued"],
        })

    ambulance_loads = {amb: 0 for amb in ambulances_fleet}
    for dispatch in dispatches:
        ambulance_loads[dispatch["ambulance"]] += 1
    ambulance_load_rows = [
        {
            "ambulance": key,
            "count": value,
            "capacity": len(mission_victims),
            "full": value >= len(mission_victims),
            "position": list(next(state["position"] for state in ambulance_states if state["name"] == key)),
        }
        for key, value in sorted(ambulance_loads.items())
    ]

    selected_victim_score_item = next((s for s in victim_scoring if s["id"] == victim.id), None)
    score_expl = selected_victim_score_item["formula"] if selected_victim_score_item else ""

    return {
        "scenario": scenario,
        "grid": mission_grid,
        "size": env.size,
        "base": env.base,
        "route_risk_weight": ROUTE_RISK_WEIGHT,
        "victims": mission_victims,
        "victim_scoring": victim_scoring,
        "hospitals": mission_hospitals,
        "presets": {k: {"name": v["name"], "badge": v["badge"], "size": v["size"], "description": v["description"]} for k, v in PRESETS.items()},
        "decision": {
            "selected_victim": victim.id,
            "selected_victims": [victim.id],
            "selected_severity": sev_ml,
            "selected_hospital": hospital,
            "selected_hospital_label": primary_dispatch["hospital_label"],
            "path_length": sum(item["distance"] for item in dispatches),
            "total_risk": sum(item["risk"] for item in dispatches),
            "saved": len(selected_ids),
            "utilization": len({item["ambulance"] for item in dispatches}) / float(len(ambulances_fleet)),
            "score_formula": score_expl,
            "reason": f"V{victim.id} has highest net priority ({score_expl}) balancing severity, proximity, and route risk.",
        },
        "ml": {
            "knn": labels[knn_pred],
            "naive_bayes": labels[nb_pred],
            "decision_tree": labels[dt_pred],
            "final": labels[final_ml],
            "final_priority": f"{labels[final_ml]} PRIORITY",
            "consensus": f"{ml_votes.count(final_ml)}/3 ({round(ml_votes.count(final_ml)/3*100)}% Consensus)",
            "features": {"distance": distance, "risk": risk, "severity": sev_ml},
            "metrics": ml_metrics,
            "explanation": f"Trained on FEMA disaster declaration dataset. All models evaluated features: Distance={distance}km, Risk={risk}, Severity={sev_ml.upper()}. Majority vote consensus: {labels[final_ml]} PRIORITY."
        },
        "fuzzy": {
            "blockage_probability": blockage_probability,
            "decision": fuzzy_result,
            "distance": distance,
            "risk": risk,
            "severity": sev_ml,
            "log": (
                f"Linguistic Inputs: Distance={distance} -> {'near' if distance<=6 else ('medium' if distance<=12 else 'far')}, "
                f"Risk={risk} -> {'low' if risk<=2 else ('medium' if risk<=5 else 'high')}, "
                f"Blockage Prob={blockage_probability} -> possible, Severity={sev_ml.upper()}\n"
                f"Fuzzy Rule: When severity is critical and hazard uncertainty is non-zero -> Output {fuzzy_result}"
            ),
            "impact": "Uncertainty dictates safety prioritization: Risk-Aware routing engaged to avoid high-hazard corridors."
        },
        "paths": {
            "base_to_victim": base_to_victim_path,
            "victim_to_hospital": hospital_path,
            "full": full_path,
            "routes": [item["path"] for item in dispatches],
        },
        "dispatches": dispatches,
        "ambulance_count": len(ambulances_fleet),
        "algorithms": algorithm_comparison,
        "analytics": {
            "comparison": analytics_rows,
            "paths": path_visualizations,
        },
        "dynamic_event": dynamic_event_record,
        "scenarios": [
            {
                "name": "Road Blockage Adaptation",
                "event": "Sudden rubble/collapse on primary street",
                "impact": "A* instantly invalidates compromised node and reroutes via safe alternate corridor.",
            },
            {
                "name": "Wildfire Hazard Detour",
                "event": "Thermal risk spiked to Level 3 along shortest path",
                "impact": "Risk-Aware A* takes a 2-step wider perimeter to incur 0 casualty risk.",
            },
            {
                "name": "New SOS Triage",
                "event": "New critical casualty reported during mission",
                "impact": "Agent dynamically re-evaluates all priorities without needing manual dispatch reassignment.",
            },
            {
                "name": "Resource Scarcity Backtracking",
                "event": "Fleet size reduced to 1 ambulance",
                "impact": "The remaining ambulance continues from each hospital to the next victim until the queue is complete.",
            },
        ],
        "assignments": assignment_rows,
        "csp": {
            "assigned_count": len(assignment),
            "unassigned_count": len(mission_victims) - len(assignment),
            "total_victims": len(mission_victims),
            "backtrack_count": backtrack_count,
            "ambulance_count_used": len([a for a, c in ambulance_loads.items() if c > 0]),
            "ambulance_loads": ambulance_load_rows,
            "all_assigned": len(assigned_ids) == len(mission_victims),
            "trace_text": csp_trace_text,
            "trace_lines": csp_trace_lines,
            "explanation": f"Risk-aware routes assign {len(assignment)} reachable victims across {len(ambulances_fleet)} ambulances. Each vehicle returns to a hospital before its next dispatch; the CSP first-wave allocation explored {backtrack_count} backtracking steps."
        },
        "kpis": {
            "victims_saved": f"{len(selected_ids)}/{len(mission_victims)}",
            "path_length": f"{sum(item['distance'] for item in dispatches)} steps",
            "total_risk": f"{sum(item['risk'] for item in dispatches)} pts",
            "fleet_util": f"{round(len({item['ambulance'] for item in dispatches})/float(len(ambulances_fleet))*100)}%",
        }
    }


def run_simulation(timeout_seconds=60):
    started_at = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"

    try:
        completed = subprocess.run(
            [sys.executable, str(SIMULATION_SCRIPT)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - started_at, 3)
        output = (exc.stdout or "") + (exc.stderr or "")
        return {
            "ok": False,
            "exit_code": None,
            "duration_seconds": duration,
            "output": output,
            "lines": output.splitlines(),
            "error": f"Simulation timed out after {timeout_seconds} seconds.",
        }

    duration = round(time.perf_counter() - started_at, 3)
    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n--- STDERR ---\n{completed.stderr}"

    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "output": output,
        "lines": output.splitlines(),
        "error": None if completed.returncode == 0 else "Simulation exited with an error.",
    }

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIDRA — Autonomous Disaster Rescue Command Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f3f6fc;
      --bg-alt: #f7f9fe;
      --panel: #ffffff;
      --panel-card: #f3f6fc;
      --panel-card-hover: #e8effb;
      --border: #dce5f4;
      --border-light: #c6d5ec;
      --text: #172342;
      --text-muted: #52617f;
      --text-dim: #6d7b96;
      --blue: #1268e8;
      --blue-glow: rgba(18, 104, 232, 0.16);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.25);
      --amber: #f59e0b;
      --amber-glow: rgba(245, 158, 11, 0.25);
      --rose: #ef4444;
      --rose-glow: rgba(239, 68, 68, 0.25);
      --indigo: #6366f1;
      --purple: #8b5cf6;
      --cell-size: 38px;
      --cell-gap: 3px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      line-height: 1.5;
      overflow-x: hidden;
    }
    a, button { color: inherit; text-decoration: none; font-family: inherit; cursor: pointer; transition: all .15s ease; border: 0; }

    /* Top Nav Bar */
    .top-header {
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 10px 20px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      gap: 16px;
    }
    .brand-wrap { display: flex; align-items: center; gap: 12px; }
    .brand-logo {
      width: 40px; height: 40px; border-radius: 10px;
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 15px var(--blue-glow); overflow: hidden;
    }
    .brand-logo img { width: 85%; height: 85%; object-fit: contain; }
    .brand-info h1 {
      font-size: 16px; font-weight: 800; letter-spacing: -0.02em;
      display: flex; align-items: center; gap: 8px;
    }
    .brand-info h1 span.pill-badge {
      background: rgba(16, 185, 129, 0.10); color: #047857;
      border: 1px solid rgba(16, 185, 129, 0.3);
      font-size: 10px; font-weight: 700; padding: 1px 7px;
      border-radius: 999px; letter-spacing: 0.05em;
    }
    .brand-info p { color: var(--text-muted); font-size: 11px; }

    .nav-tabs {
      display: flex; align-items: center; gap: 4px;
      background: #f5f8fe; padding: 4px;
      border-radius: 12px; border: 1px solid var(--border);
    }
    .nav-tab {
      padding: 7px 14px; border-radius: 8px; font-size: 12px; font-weight: 600;
      color: var(--text-muted); display: flex; align-items: center; gap: 6px;
    }
    .nav-tab:hover { color: var(--text); background: rgba(255, 255, 255, 0.05); }
    .nav-tab.active { background: var(--blue); color: #ffffff; box-shadow: 0 0 12px var(--blue-glow); }

    .header-actions { display: flex; align-items: center; gap: 10px; }
    .btn-config {
      background: linear-gradient(135deg, #1268e8, #0b57c5);
      border: 1px solid #1268e8; color: #ffffff;
      padding: 8px 16px; border-radius: 10px; font-size: 12px; font-weight: 700;
      display: flex; align-items: center; gap: 6px;
      box-shadow: 0 4px 12px rgba(18, 104, 232, 0.18);
    }
    .btn-config:hover { background: #3b82f6; border-color: #60a5fa; color: #fff; }

    /* Main Container */
    .main-container {
      max-width: 1600px; margin: 0 auto;
      padding: 16px 20px 40px; display: grid; gap: 16px;
    }

    .overview-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .overview-card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
      padding: 14px 16px; display: flex; align-items: center; justify-content: space-between;
      box-shadow: 0 4px 16px rgba(27, 55, 102, 0.06);
    }
    .overview-card span { display: block; color: var(--text-muted); font-size: 11px; font-weight: 700; }
    .overview-card strong { display: block; margin-top: 3px; color: var(--text); font-size: 23px; line-height: 1.15; }
    .overview-icon { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; background: #eaf1ff; font-size: 20px; }

    /* Top Control Bar */
    .control-deck {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 16px; padding: 12px 18px;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 14px;
      box-shadow: 0 4px 16px rgba(27, 55, 102, 0.05);
    }
    .control-group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .deck-label {
      font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
      color: var(--text-dim); text-transform: uppercase; margin-right: 4px;
    }
    .btn-ctrl {
      height: 38px; padding: 0 15px; border-radius: 9px;
      font-size: 12px; font-weight: 700; display: inline-flex; align-items: center; gap: 6px;
    }
    .btn-start {
      background: linear-gradient(135deg, #10b981, #059669); color: #fff;
      box-shadow: 0 0 12px var(--emerald-glow);
    }
    .btn-start:hover { filter: brightness(1.1); transform: scale(1.02); }
    .btn-pause { background: linear-gradient(135deg, #f59e0b, #d97706); color: #fff; }
    .btn-reset { background: var(--panel-card); border: 1px solid var(--border); color: var(--text-muted); }
    .btn-reset:hover { background: var(--panel-card-hover); color: var(--text); }

    .preset-select {
      background: var(--panel-card); border: 1px solid var(--border);
      color: var(--text); padding: 8px 12px; border-radius: 9px;
      font-size: 12px; font-weight: 600; outline: none;
    }
    .fleet-select { min-width: 126px; }
    .preset-select:focus { border-color: var(--blue); }

    .btn-roll-random {
      background: linear-gradient(135deg, #4f46e5, #7c3aed);
      color: #fff; padding: 8px 14px; border-radius: 9px;
      font-size: 12px; font-weight: 700; display: inline-flex; align-items: center; gap: 5px;
    }
    .btn-roll-random:hover { filter: brightness(1.1); }

    /* Dynamic Events */
    .events-group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .btn-event {
      background: rgba(255, 255, 255, 0.04); border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 8px; font-size: 11px; font-weight: 700;
      color: var(--text-muted); display: inline-flex; align-items: center; gap: 5px;
    }
    .btn-event:hover { background: rgba(255, 255, 255, 0.08); color: var(--text); transform: translateY(-1px); }
    .btn-event.block:hover { border-color: var(--amber); color: #fcd34d; }
    .btn-event.risk:hover { border-color: var(--rose); color: #fca5a5; }
    .btn-event.victim:hover { border-color: var(--emerald); color: #86efac; }
    .btn-event.csp:hover { border-color: var(--indigo); color: #c7d2fe; }

    /* Dynamic Event Banner */
    .event-banner {
      background: linear-gradient(90deg, #ffffff, #f3f7ff);
      border: 1px solid var(--border-light); border-left: 4px solid var(--blue);
      border-radius: 12px; padding: 14px 18px; display: none; align-items: flex-start; justify-content: space-between; gap: 16px;
    }
    .event-banner.active { display: flex; }
    .event-banner-content h4 { font-size: 13px; font-weight: 800; color: #1d4ed8; display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
    .event-banner-content p { font-size: 12px; color: var(--text-muted); }
    .event-banner-content .banner-takeaway { font-size: 11px; color: #047857; font-weight: 600; margin-top: 4px; }
    .btn-close-banner { background: transparent; color: var(--text-dim); font-size: 16px; padding: 4px; }

    .rescue-notification-stack {
      position: fixed; right: 20px; bottom: 20px; z-index: 1200;
      display: flex; flex-direction: column; gap: 8px;
      width: min(360px, calc(100vw - 32px)); pointer-events: none;
    }
    .rescue-notification {
      display: flex; align-items: flex-start; gap: 10px; padding: 12px 14px;
      background: #10251f; border: 1px solid rgba(16, 185, 129, 0.55);
      border-radius: 12px; color: #f1f5f9; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
      animation: rescueNotificationIn 0.22s ease-out;
    }
    .rescue-notification-icon {
      display: grid; place-items: center; flex: 0 0 24px; height: 24px;
      border-radius: 50%; background: #10b981; color: #052e24; font-weight: 900;
    }
    .rescue-notification strong { display: block; font-size: 12px; }
    .rescue-notification span:last-child { display: block; margin-top: 2px; color: #a7f3d0; font-size: 11px; }
    .rescue-notification.is-leaving { opacity: 0; transform: translateX(12px); transition: opacity 0.2s, transform 0.2s; }
    @keyframes rescueNotificationIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    /* Dashboard Grid */
    .dashboard-grid {
      display: grid; grid-template-columns: minmax(520px, 1fr) minmax(500px, 1.15fr);
      gap: 16px; align-items: start;
    }

    .map-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 16px; padding: 16px; display: flex; flex-direction: column; gap: 14px;
    }
    .map-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
    .map-header-title h2 { font-size: 14px; font-weight: 800; display: flex; align-items: center; gap: 8px; }
    .map-header-title span { font-size: 11px; color: var(--text-muted); }

    .algo-switcher-wrap {
      background: var(--bg-alt); border: 1px solid var(--border);
      border-radius: 12px; padding: 8px; display: flex; flex-direction: column; gap: 6px;
    }
    .algo-switcher-label {
      font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
      color: var(--text-dim); text-transform: uppercase; display: flex; align-items: center; justify-content: space-between;
    }
    .algo-buttons { display: flex; flex-wrap: wrap; gap: 5px; }
    .btn-algo {
      background: var(--panel-card); border: 1px solid var(--border);
      border-radius: 7px; padding: 5px 10px; font-size: 11px; font-weight: 700;
      color: var(--text-muted); display: flex; align-items: center; gap: 5px;
    }
    .btn-algo:hover { background: var(--panel-card-hover); color: var(--text); }
    .btn-algo.active {
      background: rgba(59, 130, 246, 0.2); border-color: var(--blue);
      color: #1d4ed8; box-shadow: 0 0 10px var(--blue-glow);
    }
    .algo-explanation-box {
      font-size: 11px; color: var(--text-muted); background: rgba(255, 255, 255, 0.02);
      border-radius: 6px; padding: 6px 10px; border-left: 2px solid var(--blue);
    }
    .algo-comparison-card {
      display: flex; flex-direction: column; gap: 8px; padding: 10px;
      background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px;
    }
    .algo-comparison-card h3 { font-size: 12px; color: var(--text); }
    .algo-comparison-card p { margin: 0; color: var(--text-muted); font-size: 11px; line-height: 1.5; }
    .algo-comparison-card .table-wrap { max-height: 220px; }

    .grid-stage {
      background: #edf2fa; border: 1px solid var(--border); border-radius: 14px;
      padding: 16px; display: flex; justify-content: center; align-items: center;
      overflow: auto; min-height: 460px;
    }
    .grid { display: grid; gap: var(--cell-gap); justify-content: center; user-select: none; }
    .cell {
      width: var(--cell-size); height: var(--cell-size); border-radius: 8px;
      background: #ffffff; border: 1px solid #dce5f4;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 800; cursor: pointer; position: relative; transition: all .15s ease;
    }
    .cell:hover { transform: scale(1.12); z-index: 10; border-color: rgba(255, 255, 255, 0.4); }
    .cell small { font-size: 8px; color: rgba(23, 35, 66, 0.4); display: none; }

    .cell.base { background: linear-gradient(135deg, #1d4ed8, #2563eb); border-color: #60a5fa; color: #fff; box-shadow: 0 0 12px var(--blue-glow); font-size: 8px; }
    .cell.hospital { background: linear-gradient(135deg, #0e7490, #0891b2); border-color: #67e8f9; color: #fff; box-shadow: 0 0 12px rgba(8, 145, 178, 0.25); font-size: 10px; }
    .cell.victim { animation: none; }
    .cell.victim-critical { background: linear-gradient(135deg, #b91c1c, #dc2626); border-color: #f87171; color: #fff; animation: pulseVictim 2s infinite ease-in-out; }
    .cell.victim-moderate { background: linear-gradient(135deg, #d97706, #f59e0b); border-color: #fcd34d; color: #172342; }
    .cell.victim-minor { background: linear-gradient(135deg, #059669, #10b981); border-color: #6ee7b7; color: #fff; }
    @keyframes pulseVictim { 0%, 100% { box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); } 50% { box-shadow: 0 0 18px rgba(239, 68, 68, 0.75); } }
    .cell.blocked { background: #334155; border-color: #64748b; color: #fff; background-image: repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(0, 0, 0, 0.18) 4px, rgba(0, 0, 0, 0.18) 8px); font-size: 8px; }
    .cell.risk-1 { background: rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.4); color: #a16207; }
    .cell.risk-2 { background: rgba(249, 115, 22, 0.18); border-color: rgba(249, 115, 22, 0.4); color: #c2410c; }
    .cell.risk-3 { background: rgba(239, 68, 68, 0.18); border-color: rgba(239, 68, 68, 0.5); color: #b91c1c; }
    .cell.path-cell { background: linear-gradient(135deg, #d97706, #f59e0b); border-color: #fde047; color: #111827; font-weight: 900; box-shadow: 0 0 12px var(--amber-glow); }
    .cell.route-marker { outline: 3px solid #f59e0b; outline-offset: -3px; }
    .cell.ambulance-active {
      background: #ffffff !important; color: #1d4ed8 !important; border: 2px solid #3b82f6 !important;
      box-shadow: 0 0 20px rgba(59, 130, 246, 0.9) !important; transform: scale(1.2) !important; z-index: 25 !important;
    }

    .map-legend { display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 14px; background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px; }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-muted); }
    .legend-dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
    .map-endpoints { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
    .map-endpoints > strong { color: var(--text); font-size: 11px; }
    .map-endpoint-chip { padding: 4px 8px; border: 1px solid var(--border); border-radius: 999px; background: var(--bg-alt); color: var(--text-muted); font-size: 10px; font-weight: 700; }
    .map-endpoint-base { color: #1d4ed8; border-color: rgba(37, 99, 235, 0.3); }
    .map-endpoint-hospital { color: #0e7490; border-color: rgba(8, 145, 178, 0.3); }
    .map-endpoint-critical { color: #b91c1c; border-color: rgba(220, 38, 38, 0.3); }
    .map-endpoint-moderate { color: #a16207; border-color: rgba(217, 119, 6, 0.3); }
    .map-endpoint-minor { color: #047857; border-color: rgba(5, 150, 105, 0.3); }
    .cell-inspector { background: var(--panel-card); border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; font-size: 11px; display: flex; align-items: center; justify-content: space-between; color: var(--text-muted); }
    .cell-inspector strong { color: var(--text); }

    /* Right Column: Explainable AI Deck */
    .ai-deck { display: flex; flex-direction: column; gap: 14px; }
    .ai-card { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .advanced-details { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
    .advanced-details summary { padding: 14px 16px; color: var(--text); cursor: pointer; font-size: 12px; font-weight: 800; }
    .advanced-details[open] summary { border-bottom: 1px solid var(--border); }
    .advanced-details > .ai-card { border: 0; border-radius: 0; box-shadow: none; }
    .ai-card-head { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
    .ai-card-title { font-size: 13px; font-weight: 800; display: flex; align-items: center; gap: 8px; }
    .badge-pill { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-blue { background: rgba(59, 130, 246, 0.10); color: #1d4ed8; border: 1px solid rgba(59, 130, 246, 0.25); }
    .badge-green { background: rgba(16, 185, 129, 0.10); color: #047857; border: 1px solid rgba(16, 185, 129, 0.25); }
    .badge-amber { background: rgba(245, 158, 11, 0.12); color: #a16207; border: 1px solid rgba(245, 158, 11, 0.28); }
    .badge-purple { background: rgba(168, 85, 247, 0.10); color: #6b21a8; border: 1px solid rgba(168, 85, 247, 0.25); }

    .formula-banner {
      background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #1d4ed8; line-height: 1.5;
    }
    .table-wrap { overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
    th { background: var(--panel-card); color: var(--text-dim); font-size: 10px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid var(--border); }
    td { padding: 9px 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); color: var(--text-muted); }
    tr:last-child td { border-bottom: 0; }
    tr.selected-row { background: rgba(59, 130, 246, 0.08); }
    tr.selected-row td { color: var(--text); font-weight: 600; }
    .sev-chip { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 800; text-transform: uppercase; }
    .sev-critical { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
    .sev-moderate { background: rgba(245, 158, 11, 0.15); color: #a16207; }
    .sev-minor { background: rgba(16, 185, 129, 0.12); color: #047857; }

    .ml-vote-grid { display: grid; grid-template-columns: repeat(3, 1fr) 1.2fr; gap: 8px; }
    .ml-box { background: var(--panel-card); border: 1px solid var(--border); border-radius: 10px; padding: 10px; text-align: center; }
    .ml-box span { font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 700; display: block; margin-bottom: 3px; }
    .ml-box strong { font-size: 13px; color: var(--text); }
    .ml-box.consensus { background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(124, 58, 237, 0.15)); border-color: rgba(99, 102, 241, 0.4); }
    .ml-box.consensus strong { color: #4f46e5; font-weight: 800; }

    .fuzzy-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
    .fuzzy-metric { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px; padding: 8px; text-align: center; }
    .fuzzy-metric span { font-size: 10px; color: var(--text-dim); display: block; }
    .fuzzy-metric strong { font-size: 12px; color: #a16207; }

    .csp-board { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .ambulance-card { background: var(--panel-card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
    .amb-head { display: flex; align-items: center; justify-content: space-between; font-size: 12px; font-weight: 800; }
    .amb-victims-list { display: flex; flex-direction: column; gap: 4px; }
    .amb-location { color: var(--text-muted); font-size: 10px; }
    .amb-victim-chip { background: var(--bg-alt); border-radius: 6px; padding: 5px 8px; font-size: 11px; display: flex; align-items: center; justify-content: space-between; }
    .capacity-meter { background: var(--border); height: 6px; border-radius: 999px; overflow: hidden; }
    .capacity-bar { height: 100%; background: linear-gradient(90deg, #10b981, #3b82f6); }

    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .kpi-card { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px; padding: 10px; text-align: center; }
    .kpi-card span { font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 700; display: block; margin-bottom: 2px; }
    .kpi-card strong { font-size: 15px; font-weight: 800; color: var(--text); }

    .mission-log {
      background: #f7f9fe; border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #52617f; max-height: 90px; overflow-y: auto; line-height: 1.5;
    }

    /* Modals */
    .modal {
      display: none; position: fixed; inset: 0; background: rgba(7, 11, 20, 0.85);
      backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; padding: 20px;
    }
    .modal.active { display: flex; }
    .modal-sheet {
      background: var(--panel); border: 1px solid var(--border-light); border-radius: 20px;
      max-width: 1050px; width: 100%; max-height: 92vh; display: flex; flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7); overflow: hidden;
    }
    .modal-sheet-head {
      padding: 18px 24px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
    }
    .modal-sheet-head h3 { font-size: 17px; font-weight: 800; display: flex; align-items: center; gap: 8px; }
    .modal-sheet-head p { font-size: 12px; color: var(--text-muted); }
    .btn-close-modal {
      background: var(--panel-card); border: 1px solid var(--border);
      color: var(--text-muted); border-radius: 8px; padding: 6px 12px; font-size: 12px; font-weight: 700;
    }
    .btn-close-modal:hover { color: var(--text); background: var(--panel-card-hover); }
    .modal-sheet-body { padding: 20px 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }

    .guide-tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 12px; flex-wrap: wrap; }
    .guide-tab-btn {
      padding: 8px 14px; border-radius: 8px; font-size: 12px; font-weight: 700;
      background: var(--bg-alt); border: 1px solid var(--border); color: var(--text-muted);
    }
    .guide-tab-btn.active { background: var(--blue); border-color: var(--blue); color: #fff; }
    .guide-tab-pane { display: none; flex-direction: column; gap: 14px; }
    .guide-tab-pane.active { display: flex; }
    .guide-callout { background: var(--panel-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 8px; }
    .guide-callout h4 { font-size: 13px; font-weight: 800; color: #1d4ed8; display: flex; align-items: center; gap: 8px; }
    .guide-callout p, .guide-callout li { font-size: 12px; color: var(--text-muted); line-height: 1.6; }
    .speech-box { background: rgba(59, 130, 246, 0.08); border-left: 3px solid var(--blue); border-radius: 8px; padding: 12px 16px; font-style: italic; color: #e2e8f0; font-size: 12.5px; line-height: 1.6; }

    /* Scenario Studio Specifics */
    .studio-palette {
      display: flex; flex-wrap: wrap; gap: 6px; background: var(--bg-alt);
      border: 1px solid var(--border); border-radius: 12px; padding: 10px;
    }
    .btn-tool {
      background: var(--panel-card); border: 1px solid var(--border); border-radius: 8px;
      padding: 6px 12px; font-size: 11px; font-weight: 700; color: var(--text-muted);
      display: flex; align-items: center; gap: 6px;
    }
    .btn-tool:hover { background: var(--panel-card-hover); color: var(--text); }
    .btn-tool.active {
      background: rgba(59, 130, 246, 0.12); border-color: var(--blue); color: #1d4ed8; box-shadow: 0 0 10px var(--blue-glow);
    }
    .painter-canvas-wrap {
      background: #edf2fa; border: 1px solid var(--border); border-radius: 14px;
      padding: 16px; display: flex; justify-content: center; align-items: center; overflow: auto; min-height: 380px;
    }
    .studio-stats-bar {
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
      background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; font-size: 12px; color: var(--text-muted);
    }
    #studioStatsTally { flex: 1; min-width: 260px; line-height: 1.6; }
    .preset-card-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;
    }
    .preset-card {
      background: var(--panel-card); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px; display: flex; flex-direction: column; justify-content: space-between; gap: 10px;
      transition: all .2s;
    }
    .preset-card:hover { border-color: var(--blue); transform: translateY(-2px); }
    .preset-card h5 { font-size: 13px; font-weight: 800; color: var(--text); margin-bottom: 2px; }
    .preset-card p { font-size: 11px; color: var(--text-muted); }
    .btn-load-preset {
      background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3);
      color: #1d4ed8; padding: 6px 12px; border-radius: 7px; font-size: 11px; font-weight: 700; width: 100%;
    }
    .btn-load-preset:hover { background: var(--blue); color: #fff; }

    .json-editor {
      background: #070b14; border: 1px solid var(--border); border-radius: 10px;
      padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #a5f3fc;
      width: 100%; height: 280px; resize: vertical; outline: none; line-height: 1.5;
    }
    .json-editor:focus { border-color: var(--blue); }

    .loader-overlay {
      display: none; position: fixed; inset: 0; background: rgba(11, 17, 32, 0.7);
      backdrop-filter: blur(6px); z-index: 2000; align-items: center; justify-content: center;
    }
    .loader-overlay.active { display: flex; }
    .loader-card {
      background: var(--panel); border: 1px solid var(--border-light); border-radius: 16px;
      padding: 24px 32px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 12px;
    }
    .spinner {
      width: 44px; height: 44px; border: 3px solid rgba(59, 130, 246, 0.2);
      border-top-color: var(--blue); border-radius: 50%; animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 1150px) {
      .dashboard-grid { grid-template-columns: 1fr; }
      .top-header { flex-wrap: wrap; }
      .nav-tabs { order: 3; width: 100%; overflow-x: auto; }
    }
    @media (max-width: 720px) {
      .overview-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .main-container { padding: 12px 10px 24px; }
      .top-header { padding: 10px; }
      .map-card, .ai-card { padding: 12px; }
      .grid-stage { min-height: 340px; padding: 10px; }
    }
  </style>
</head>
<body>
  <header class="top-header">
    <div class="brand-wrap">
      <div class="brand-logo">
        <img src="/logo.png" alt="AIDRA Logo" onerror="this.onerror=null; this.parentElement.innerHTML='🚑';">
      </div>
      <div class="brand-info">
        <h1>AIDRA <span class="pill-badge">AI Active</span></h1>
        <p>Adaptive Intelligent Disaster Response Agent — Mission Control</p>
      </div>
    </div>

    <nav class="nav-tabs">
      <a class="nav-tab active" href="/">Overview</a>
      <a class="nav-tab" href="/search-trace">Route algorithms</a>
      <a class="nav-tab" href="/csp-solver">Fleet planning</a>
      <a class="nav-tab" href="/ml-studio">ML Studio</a>
      <a class="nav-tab" href="/analytics">Analytics</a>
      <a class="nav-tab" href="/simulation">Simulation log</a>
    </nav>

    <div class="header-actions">
      <button class="btn-config" id="openConfigBtn" type="button">
        Edit disaster map
      </button>
    </div>
  </header>

  <main class="main-container">
    <section class="overview-stats" aria-label="Mission overview">
      <div class="overview-card"><div><span>Victims reported</span><strong id="statVictims">{{ data.victims|length }}</strong></div><div class="overview-icon">🆘</div></div>
      <div class="overview-card"><div><span>Victims in rescue plan</span><strong id="statRescued">{{ data.decision.saved }}</strong></div><div class="overview-icon">✅</div></div>
      <div class="overview-card"><div><span>Ambulances available</span><strong id="statAmbulances">{{ data.ambulance_count }}</strong></div><div class="overview-icon">🚑</div></div>
      <div class="overview-card"><div><span>Route distance</span><strong id="statDistance">{{ data.decision.path_length }} steps</strong></div><div class="overview-icon">🛣️</div></div>
    </section>

    <!-- Top Action Deck -->
    <section class="control-deck">
      <div class="control-group">
        <button class="btn-ctrl btn-start" id="startBtn" type="button">▶ Run rescue plan</button>
        <button class="btn-ctrl btn-pause" id="pauseBtn" type="button">⏸ Pause</button>
        <button class="btn-ctrl btn-reset" id="resetBtn" type="button">↺ Reset map</button>
        <label class="deck-label" for="ambulanceCountSelect">Fleet size</label>
        <select class="preset-select fleet-select" id="ambulanceCountSelect" aria-label="Number of ambulances">
          {% for count in range(1, 7) %}
            <option value="{{ count }}" {{ 'selected' if count == data.ambulance_count else '' }}>{{ count }} ambulance{{ '' if count == 1 else 's' }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="control-group">
        <select class="preset-select" id="presetSelect" aria-label="Select Scenario Preset">
          {% for key, p in data.presets.items() %}
            <option value="{{ key }}">{{ p.badge }} — {{ p.name }} ({{ p.size }}x{{ p.size }})</option>
          {% endfor %}
        </select>
        <button class="btn-roll-random" id="quickRollRandomBtn" type="button" title="Generate procedural random disaster map">
          🎲 Random map
        </button>
      </div>

      <div class="control-group events-group">
        <span class="deck-label">Test:</span>
        <button class="btn-event block" id="evBlockBtn" type="button">🚧 Block road</button>
        <button class="btn-event risk" id="evRiskBtn" type="button">🔥 Add hazard</button>
        <button class="btn-event victim" id="evVictimBtn" type="button">🚨 New victim</button>
        <button class="btn-event csp" id="evCspBtn" type="button">🚑 Remove ambulance</button>
      </div>
    </section>

    <!-- Dynamic Event Alert Banner -->
    <section class="event-banner" id="eventBanner">
      <div class="event-banner-content">
        <h4 id="eventTitle">🚧 Dynamic Replan Active</h4>
        <p id="eventDescription">Environmental change detected. Recomputing AI decisions...</p>
        <div class="banner-takeaway" id="eventTakeaway">✓ AI adapted autonomously without mission failure.</div>
      </div>
      <button class="btn-close-banner" id="closeBannerBtn" type="button">✕</button>
    </section>

    <div class="rescue-notification-stack" id="rescueNotifications" aria-live="polite" aria-relevant="additions"></div>

    <!-- Dashboard Main Grid -->
    <section class="dashboard-grid">
      <!-- Left Column: Map & Search Algorithm Switcher -->
      <section class="map-card">
        <div class="map-header">
          <div class="map-header-title">
            <h2>Disaster map <span class="badge-pill badge-blue">{{ data.size }} × {{ data.size }}</span></h2>
            <span>Hover over a cell to inspect road and hazard details</span>
          </div>
          <div class="cell-inspector" id="cellInspector">
            <span>Inspector: <strong>Hover or click any cell</strong></span>
          </div>
        </div>

        <div class="algo-switcher-wrap">
          <div class="algo-switcher-label">
            <span>Preview route algorithms:</span>
            <span id="activeAlgoBadge" class="badge-pill badge-green">Risk-Aware A* Active</span>
          </div>
          <div class="algo-buttons" id="algoButtons">
            {% for algo in data.algorithms %}
              <button class="btn-algo {{ 'active' if loop.first else '' }}" data-algo="{{ algo.name }}" type="button">
                {{ algo.name }} ({{ algo.length }} steps, {{ algo.risk }} risk)
              </button>
            {% endfor %}
          </div>
          <div class="algo-explanation-box" id="algoExplanation">
            Risk-aware A* weighs travel distance and hazard exposure. It prefers safer roads but cannot guarantee zero risk.
          </div>
          <section class="algo-comparison-card" aria-label="Algorithm route comparison">
            <h3>Route comparison for Victim V{{ data.decision.selected_victim }}</h3>
            <p id="algoPerformanceSummary"></p>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr><th>Algorithm</th><th>Steps</th><th>Risk</th><th>Weighted cost</th><th>vs Risk-Aware A*</th></tr>
                </thead>
                <tbody id="algorithmComparisonRows"></tbody>
              </table>
            </div>
            <p>Weighted cost = steps + {{ data.route_risk_weight }} × risk points; lower is better. This compares route outcomes, not computation time.</p>
          </section>
        </div>

        <div class="grid-stage">
          <div class="grid" id="gridMap" style="grid-template-columns: repeat({{ data.size }}, var(--cell-size));">
            {% for row in data.grid %}
              {% for cell in row %}
                <div class="cell {{ cell.terrain_type }} {{ 'victim-' ~ cell.victim_severity if cell.victim_severity else '' }}"
                     data-x="{{ cell.x }}" data-y="{{ cell.y }}"
                     data-risk="{{ cell.risk }}" data-type="{{ cell.terrain_type }}"
                     title="({{ cell.x }}, {{ cell.y }}) - {{ cell.terrain_label }} (Risk: {{ cell.risk }})">
                  {{ cell.marker_label or cell.icon or cell.terrain_label }}
                  <small>{{ cell.x }},{{ cell.y }}</small>
                </div>
              {% endfor %}
            {% endfor %}
          </div>
        </div>

        <div class="map-legend">
          <div class="legend-item"><span class="legend-dot" style="background:#2563eb"></span>Base (BASE)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#0891b2"></span>Hospital (H#)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>Critical victim (V#)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>Moderate victim (V#)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#10b981"></span>Minor victim (V#)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#475569"></span>Blocked road (BLOCK)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>Fire L1</div>
          <div class="legend-item"><span class="legend-dot" style="background:#f97316"></span>Fire L2</div>
          <div class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>Fire L3</div>
          <div class="legend-item"><span class="legend-dot" style="background:#d97706"></span>Rescue route</div>
        </div>
        <div class="map-endpoints" aria-label="Map endpoint locations">
          <strong>Map endpoints:</strong>
          <span class="map-endpoint-chip map-endpoint-base">Base BASE · ({{ data.base[0] }}, {{ data.base[1] }})</span>
          {% for hospital in data.hospitals %}
            <span class="map-endpoint-chip map-endpoint-hospital">Hospital {{ hospital.label }} · ({{ hospital.x }}, {{ hospital.y }})</span>
          {% endfor %}
          {% for victim in data.victims %}
            <span class="map-endpoint-chip map-endpoint-{{ victim.severity }}">Victim V{{ victim.id }} · {{ victim.severity|upper }} · ({{ victim.x }}, {{ victim.y }})</span>
          {% endfor %}
        </div>
      </section>

      <!-- Right Column: Explainable AI Decision Deck -->
      <aside class="ai-deck">
        <section class="ai-card">
          <div class="ai-card-head">
            <h3 class="ai-card-title">Victim status & rescue priority</h3>
            <span class="badge-pill badge-blue">Automatic triage</span>
          </div>
          <div class="formula-banner">
            Priority Score = Severity Score (Crit=100, Mod=50, Min=10) - (Distance + Path Risk)
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Rank</th><th>ID</th><th>Severity</th><th>Distance</th><th>Risk</th><th>Score Formula</th><th>Status</th></tr>
              </thead>
              <tbody id="victimScoringRows">
                {% for v in data.victim_scoring %}
                  <tr class="{{ 'selected-row' if v.selected else '' }}" data-victim-id="{{ v.id }}">
                    <td>#{{ v.rank }}</td>
                    <td><strong>V{{ v.id }}</strong></td>
                    <td><span class="sev-chip sev-{{ v.severity }}">{{ v.severity|upper }}</span></td>
                    <td>{{ v.distance }} steps</td>
                    <td>{{ v.risk }} pts</td>
                    <td><code>{{ v.formula }}</code></td>
                    <td>{% if v.rescued %}<span class="badge-pill badge-green victim-status">Planned</span>{% else %}<span class="badge-pill badge-amber victim-status">Waiting</span>{% endif %}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          <p style="font-size: 11px; color: var(--text-muted);" id="victimDecisionExplanation">
            {{ data.decision.reason }}
          </p>
        </section>

        <details class="advanced-details">
          <summary>AI prediction details (machine learning & fuzzy analysis)</summary>
          <section class="ai-card">
          <div class="ai-card-head">
            <h3 class="ai-card-title">AI decision support</h3>
            <span class="badge-pill badge-purple">Trained on FEMA Data</span>
          </div>
          <div class="ml-vote-grid" id="mlVoteGrid">
            <div class="ml-box"><span>k-NN Model</span><strong id="knnVote">{{ data.ml.knn }}</strong></div>
            <div class="ml-box"><span>Naive Bayes</span><strong id="nbVote">{{ data.ml.naive_bayes }}</strong></div>
            <div class="ml-box"><span>Decision Tree</span><strong id="dtVote">{{ data.ml.decision_tree }}</strong></div>
            <div class="ml-box consensus"><span>Ensemble Majority</span><strong id="consensusVote">{{ data.ml.final }} PRIORITY</strong></div>
          </div>
          <div class="fuzzy-grid" id="fuzzyGrid">
            <div class="fuzzy-metric"><span>Distance</span><strong id="fzDist">{{ data.fuzzy.distance }} steps</strong></div>
            <div class="fuzzy-metric"><span>Route Hazard</span><strong id="fzRisk">{{ data.fuzzy.risk }} pts</strong></div>
            <div class="fuzzy-metric"><span>Blockage Prob</span><strong id="fzProb">{{ (data.fuzzy.blockage_probability * 100)|int }}%</strong></div>
            <div class="fuzzy-metric"><span>Fuzzy Output</span><strong id="fzDecision">{{ data.fuzzy.decision }}</strong></div>
          </div>
          <p style="font-size: 11px; color: var(--text-muted);" id="fuzzyImpactText">
            💡 <strong>Uncertainty Reason:</strong> {{ data.fuzzy.impact }}
          </p>
          </section>
        </details>

        <section class="ai-card">
          <div class="ai-card-head">
            <h3 class="ai-card-title">Ambulance fleet & rescue plan</h3>
            <span class="badge-pill badge-amber">Round-trip dispatch</span>
          </div>
          <div class="csp-board" id="cspBoard">
            {% for load in data.csp.ambulance_loads %}
              <div class="ambulance-card">
                <div class="amb-head">
                  <span>🚑 {{ load.ambulance }}</span>
                  <span class="badge-pill {{ 'badge-green' if load.count > 0 else 'badge-blue' }} ambulance-rescue-count" data-count="{{ load.count }}">{{ load.count }} planned</span>
                </div>
                <div class="amb-location">Current location: ({{ load.position[0] }}, {{ load.position[1] }})</div>
                <div class="capacity-meter">
                  <div class="capacity-bar" style="width: {{ (load.count / load.capacity * 100)|int }}%;"></div>
                </div>
                <div class="amb-victims-list">
                  {% for item in data.assignments if item.ambulance == load.ambulance %}
                    <div class="amb-victim-chip">
                      <span><strong>Victim V{{ item.victim }}</strong> ({{ item.location }})</span>
                      <span class="sev-chip sev-{{ item.severity }}">{{ item.severity }}</span>
                    </div>
                  {% endfor %}
                </div>
              </div>
            {% endfor %}
          </div>
          <p style="font-size: 11px; color: var(--text-muted);" id="cspExplanation">
            Each ambulance returns to a hospital before taking another rescue call. Select a fleet size above to dispatch more vehicles.
          </p>
        </section>

        <section class="ai-card">
          <div class="ai-card-head">
            <h3 class="ai-card-title">Mission summary & activity</h3>
            <span class="badge-pill badge-green">Operational</span>
          </div>
          <div class="kpi-row" id="kpiDeck">
            <div class="kpi-card"><span>Planned rescues</span><strong id="kpiSaved">{{ data.kpis.victims_saved }}</strong></div>
            <div class="kpi-card"><span>Path Steps</span><strong id="kpiLength">{{ data.kpis.path_length }}</strong></div>
            <div class="kpi-card"><span>Risk Incurred</span><strong id="kpiRisk" style="color:#047857;">{{ data.kpis.total_risk }}</strong></div>
            <div class="kpi-card"><span>Fleet Util</span><strong id="kpiUtil">{{ data.kpis.fleet_util }}</strong></div>
          </div>
          <div class="mission-log" id="eventLog">
            [00:00:00] AIDRA Mission Control initialized. Ready to execute rescue plan.
          </div>
        </section>
      </aside>
    </section>
  </main>

  <!-- Ultra-Flexible Scenario Studio & Map Architect Modal -->
  <div class="modal" id="configModal" aria-hidden="true">
    <div class="modal-sheet" style="max-width: 1100px;" role="dialog" aria-modal="true" aria-labelledby="studioTitle">
      <div class="modal-sheet-head">
        <div>
          <h3 id="studioTitle">⚙️ Scenario Studio & Map Architect</h3>
          <p>Visually paint, randomly generate, load diverse presets, or edit JSON to create unlimited custom disaster scenarios.</p>
        </div>
        <button class="btn-close-modal" id="closeConfigBtn" type="button">✕ Close</button>
      </div>

      <div class="modal-sheet-body">
        <div class="guide-tabs" id="studioTabs">
          <button class="guide-tab-btn active" data-tab="painter" type="button">🎨 Visual Grid Painter</button>
          <button class="guide-tab-btn" data-tab="generator" type="button">🎲 Procedural Generator</button>
          <button class="guide-tab-btn" data-tab="presets" type="button">📚 8 Disaster Templates</button>
          <button class="guide-tab-btn" data-tab="json" type="button">💻 JSON Code Studio</button>
        </div>

        <!-- Mode 1: Visual Grid Painter -->
        <div class="guide-tab-pane active" id="tab-painter">
          <div class="studio-palette">
            <span style="font-size:11px; font-weight:800; color:var(--text-dim); text-transform:uppercase; margin-right:4px; display:flex; align-items:center;">Tool:</span>
            <button class="btn-tool active" data-tool="base" type="button">🚑 Set Base</button>
            <button class="btn-tool" data-tool="hospital" type="button">🏥 Hospital</button>
            <button class="btn-tool" data-tool="victim_critical" type="button">🚨 Critical V</button>
            <button class="btn-tool" data-tool="victim_moderate" type="button">⚠️ Moderate V</button>
            <button class="btn-tool" data-tool="victim_minor" type="button">🟢 Minor V</button>
            <button class="btn-tool" data-tool="block" type="button">🚧 Roadblock</button>
            <button class="btn-tool" data-tool="risk_1" type="button">🔥 Fire L1</button>
            <button class="btn-tool" data-tool="risk_2" type="button">🔥 Fire L2</button>
            <button class="btn-tool" data-tool="risk_3" type="button">🔥 Fire L3</button>
            <button class="btn-tool" data-tool="erase" type="button" style="color:#f87171;">🧹 Eraser</button>
          </div>

          <div class="studio-stats-bar">
            <div>
              Grid Size:
              <select id="studioSizeSelect" style="background:var(--panel-card); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:3px 8px; font-size:11px; margin-left:4px;">
                <option value="8">8x8</option>
                <option value="10" selected>10x10</option>
                <option value="12">12x12</option>
                <option value="14">14x14</option>
                <option value="16">16x16</option>
              </select>
            </div>
            <div id="studioStatsTally">
              🚑 Base: <strong>(0,0)</strong> &bull; 🏥 Hosp: <strong>2</strong> &bull; 🆘 Casualties: <strong>4</strong> &bull; 🚧 Blocks: <strong>4</strong> &bull; 🔥 Hazards: <strong>6</strong>
            </div>
            <div style="display:flex; gap:6px;">
              <button id="studioClearObstaclesBtn" type="button" style="background:var(--panel-card); border:1px solid var(--border); color:var(--text-muted); padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700;">🧹 Clear Hazards</button>
              <button id="studioResetBtn" type="button" style="background:var(--panel-card); border:1px solid var(--border); color:var(--text-muted); padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700;">↺ Current</button>
            </div>
          </div>

          <div class="painter-canvas-wrap">
            <div class="grid" id="painterGrid" style="grid-template-columns: repeat(10, 32px);">
              <!-- Rendered via JS -->
            </div>
          </div>
          <div class="map-legend" aria-label="Map marker color legend">
            <div class="legend-item"><span class="legend-dot" style="background:#2563eb"></span>Base (BASE)</div>
            <div class="legend-item"><span class="legend-dot" style="background:#0891b2"></span>Hospital (H#)</div>
            <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>Critical victim (V#)</div>
            <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>Moderate victim (V#)</div>
            <div class="legend-item"><span class="legend-dot" style="background:#10b981"></span>Minor victim (V#)</div>
            <div class="legend-item"><span class="legend-dot" style="background:#475569"></span>Blocked road (BLOCK)</div>
            <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>Fire L1</div>
            <div class="legend-item"><span class="legend-dot" style="background:#f97316"></span>Fire L2</div>
            <div class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>Fire L3</div>
          </div>

          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:11px; color:var(--text-dim);">Select a tool above, then click on any cell to place, move, or erase items.</span>
            <button id="studioApplyBtn" type="button" style="background:linear-gradient(135deg, #10b981, #059669); color:#fff; font-weight:800; padding:10px 24px; border-radius:10px; font-size:13px; box-shadow:0 0 15px var(--emerald-glow);">
              ⚡ Apply Scenario & Replan Mission
            </button>
          </div>
        </div>

        <!-- Mode 2: Procedural Generator -->
        <div class="guide-tab-pane" id="tab-generator">
          <div class="guide-callout">
            <h4>🎲 Procedural Scenario Generator</h4>
            <p>Generate mathematically guaranteed reachable disaster environments with randomized road blockages and fire clusters.</p>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-top:8px;">
              <div>
                <label style="font-size:10px; color:var(--text-dim); text-transform:uppercase; font-weight:700; display:block; margin-bottom:4px;">Grid Size</label>
                <select id="genSize" style="background:var(--bg-alt); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:8px 12px; width:100%; font-size:12px;">
                  <option value="8">8x8 (Compact)</option>
                  <option value="10" selected>10x10 (Standard)</option>
                  <option value="12">12x12 (Expanded)</option>
                  <option value="14">14x14 (Megacity)</option>
                </select>
              </div>
              <div>
                <label style="font-size:10px; color:var(--text-dim); text-transform:uppercase; font-weight:700; display:block; margin-bottom:4px;">Casualties Count</label>
                <select id="genVictims" style="background:var(--bg-alt); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:8px 12px; width:100%; font-size:12px;">
                  <option value="2">2 Victims</option>
                  <option value="4" selected>4 Victims</option>
                  <option value="6">6 Victims (CSP Stress)</option>
                  <option value="8">8 Victims (High Triage)</option>
                </select>
              </div>
              <div>
                <label style="font-size:10px; color:var(--text-dim); text-transform:uppercase; font-weight:700; display:block; margin-bottom:4px;">Hospitals</label>
                <select id="genHospitals" style="background:var(--bg-alt); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:8px 12px; width:100%; font-size:12px;">
                  <option value="1">1 Hospital</option>
                  <option value="2" selected>2 Hospitals</option>
                  <option value="3">3 Hospitals</option>
                </select>
              </div>
              <div>
                <label style="font-size:10px; color:var(--text-dim); text-transform:uppercase; font-weight:700; display:block; margin-bottom:4px;">Fire Hazards</label>
                <select id="genHazards" style="background:var(--bg-alt); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:8px 12px; width:100%; font-size:12px;">
                  <option value="low">Low Hazards</option>
                  <option value="medium" selected>Moderate Clusters</option>
                  <option value="high">Extreme Firestorms</option>
                </select>
              </div>
              <div>
                <label style="font-size:10px; color:var(--text-dim); text-transform:uppercase; font-weight:700; display:block; margin-bottom:4px;">Roadblock Density</label>
                <select id="genBlocks" style="background:var(--bg-alt); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:8px 12px; width:100%; font-size:12px;">
                  <option value="sparse">Sparse Obstacles</option>
                  <option value="medium" selected>Balanced Rubble</option>
                  <option value="heavy">Dense Rubble Labyrinth</option>
                </select>
              </div>
            </div>

            <button id="studioGenerateBtn" type="button" style="background:linear-gradient(135deg, #4f46e5, #7c3aed); color:#fff; font-weight:800; padding:12px 20px; border-radius:10px; font-size:13px; margin-top:14px; width:100%;">
              🎲 Generate & Preview Valid Scenario
            </button>
          </div>
        </div>

        <!-- Mode 3: 8 Preset Templates -->
        <div class="guide-tab-pane" id="tab-presets">
          <div class="preset-card-grid">
            {% for key, p in data.presets.items() %}
              <div class="preset-card">
                <div>
                  <span class="badge-pill badge-blue" style="margin-bottom:6px; display:inline-block;">{{ p.badge }}</span>
                  <h5>{{ p.name }}</h5>
                  <p>{{ p.description }}</p>
                </div>
                <button class="btn-load-preset" data-preset="{{ key }}" type="button">
                  📥 Load Preset ({{ p.size }}x{{ p.size }})
                </button>
              </div>
            {% endfor %}
          </div>
        </div>

        <!-- Mode 4: JSON Code Studio -->
        <div class="guide-tab-pane" id="tab-json">
          <div class="guide-callout">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <h4>💻 Raw JSON Scenario Code</h4>
              <div style="display:flex; gap:8px;">
                <button id="copyJsonBtn" type="button" style="background:var(--panel-card); border:1px solid var(--border); color:var(--text); padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700;">📋 Copy JSON</button>
                <button id="formatJsonBtn" type="button" style="background:var(--panel-card); border:1px solid var(--border); color:var(--text); padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700;">⚡ Format</button>
              </div>
            </div>
            <textarea class="json-editor" id="jsonScenarioEditor" spellcheck="false"></textarea>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
              <span id="jsonErrorStatus" style="font-size:11px; color:#f87171;"></span>
              <button id="studioImportJsonBtn" type="button" style="background:var(--blue); color:#fff; font-weight:800; padding:10px 20px; border-radius:9px; font-size:12px;">
                📥 Import & Load JSON to Canvas
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Loading Animation Overlay -->
  <div class="loader-overlay" id="loadingOverlay">
    <div class="loader-card">
      <div class="spinner"></div>
      <strong id="loadingTitle" style="font-size:15px; color:var(--text);">Computing Autonomous Mission Plan</strong>
      <span id="loadingMessage" style="font-size:12px; color:var(--text-muted);">Running A* search, CSP MRV solver, and ML consensus...</span>
    </div>
  </div>

  <script>
    const initialData = {{ data|tojson }};
    let currentData = initialData;
    let selectedAlgo = "Risk-Aware A*";
    let isPaused = false;
    let animationTimer = null;
    let routeCoords = [];
    let routeStepIndex = 0;
    let rescuedVictimIds = new Set();

    // Studio Canvas state
    let studioScenario = JSON.parse(JSON.stringify(currentData.scenario));
    let activePaintTool = "base";

    // Elements
    const gridMap = document.getElementById("gridMap");
    const eventBanner = document.getElementById("eventBanner");
    const eventTitle = document.getElementById("eventTitle");
    const eventDescription = document.getElementById("eventDescription");
    const eventTakeaway = document.getElementById("eventTakeaway");
    const closeBannerBtn = document.getElementById("closeBannerBtn");
    const eventLog = document.getElementById("eventLog");
    const cellInspector = document.getElementById("cellInspector");
    const configModal = document.getElementById("configModal");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const startBtn = document.getElementById("startBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const resetBtn = document.getElementById("resetBtn");
    const ambulanceCountSelect = document.getElementById("ambulanceCountSelect");
    const presetSelect = document.getElementById("presetSelect");
    const quickRollRandomBtn = document.getElementById("quickRollRandomBtn");

    function log(msg) {
      const now = new Date().toTimeString().split(" ")[0];
      eventLog.innerHTML = `[${now}] ${msg}<br>` + eventLog.innerHTML;
    }

    function notifyVictimRescued(victimId) {
      const notification = document.createElement("div");
      notification.className = "rescue-notification";
      notification.setAttribute("role", "status");

      const icon = document.createElement("span");
      icon.className = "rescue-notification-icon";
      icon.textContent = "✓";

      const message = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `Victim V${victimId} rescued`;
      const detail = document.createElement("span");
      detail.textContent = "Ambulance is transporting the victim to a hospital.";
      message.append(title, detail);
      notification.append(icon, message);
      document.getElementById("rescueNotifications").append(notification);

      setTimeout(() => {
        notification.classList.add("is-leaving");
        setTimeout(() => notification.remove(), 220);
      }, 5000);
    }

    // Modal toggles
    document.getElementById("openConfigBtn").addEventListener("click", () => {
      studioScenario = JSON.parse(JSON.stringify(currentData.scenario));
      syncStudioUi();
      configModal.classList.add("active");
    });
    document.getElementById("closeConfigBtn").addEventListener("click", () => configModal.classList.remove("active"));
    closeBannerBtn.addEventListener("click", () => eventBanner.classList.remove("active"));

    // Studio Tabs
    document.querySelectorAll("#studioTabs .guide-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#studioTabs .guide-tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll("#configModal .guide-tab-pane").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
        if (btn.dataset.tab === "json") {
          document.getElementById("jsonScenarioEditor").value = JSON.stringify(studioScenario, null, 2);
        }
      });
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        configModal.classList.remove("active");
      }
    });

    // Grid rendering
    function renderGrid(data, activeRoute = [], ambulanceCoord = null, hiddenVictimIds = rescuedVictimIds) {
      const routeSet = new Set(activeRoute.map(p => `${p[0]},${p[1]}`));
      const victimsByCell = new Map(data.victims.map(victim => [`${victim.x},${victim.y}`, victim]));
      gridMap.style.gridTemplateColumns = `repeat(${data.size}, var(--cell-size))`;

      let html = "";
      for (let x = 0; x < data.size; x++) {
        for (let y = 0; y < data.size; y++) {
          const cell = data.grid[x][y];
          const victim = victimsByCell.get(`${x},${y}`);
          const victimRescued = cell.terrain_type === "victim" && victim && hiddenVictimIds.has(victim.id);
          const terrainType = victimRescued ? (cell.risk > 0 ? `risk risk-${cell.risk}` : "empty") : cell.terrain_type;
          const terrainLabel = victimRescued ? (cell.risk > 0 ? `Fire hazard Level ${cell.risk}` : `Victim V${victim.id} rescued`) : cell.terrain_label;
          const markerLabel = victimRescued ? (cell.risk > 0 ? `L${cell.risk}` : "") : (cell.marker_label || cell.icon || cell.terrain_label || "");
          const isAmb = ambulanceCoord && ambulanceCoord[0] === x && ambulanceCoord[1] === y;
          const onPath = routeSet.has(`${x},${y}`);

          let cellClass = `cell ${terrainType}`;
          if (terrainType === "victim" && victim) cellClass += ` victim-${victim.severity}`;
          let label = markerLabel;

          if (isAmb) {
            cellClass += " ambulance-active";
            label = "🚑";
          } else if (onPath) {
            if (["base", "victim", "hospital"].includes(terrainType)) {
              cellClass += " route-marker";
            } else {
              cellClass = "cell path-cell";
              label = "•";
            }
          }

          html += `
            <div class="${cellClass}"
                 data-x="${x}" data-y="${y}"
                 data-risk="${cell.risk}"
                 data-type="${terrainType}"
                 title="(${x}, ${y}) - ${terrainLabel} (Risk: ${cell.risk})">
              ${label}
              <small>${x},${y}</small>
            </div>
          `;
        }
      }
      gridMap.innerHTML = html;

      gridMap.querySelectorAll(".cell").forEach(c => {
        c.addEventListener("mouseenter", () => {
          const x = c.dataset.x, y = c.dataset.y, risk = c.dataset.risk, type = c.dataset.type;
          cellInspector.innerHTML = `Inspector: <strong>(${x}, ${y})</strong> &bull; Type: <strong>${type.toUpperCase()}</strong> &bull; Risk Level: <strong>${risk}</strong>`;
        });
      });
    }

    function updatePanels(data) {
      const victimRows = data.victim_scoring.map(v => `
        <tr class="${v.selected ? 'selected-row' : ''}" data-victim-id="${v.id}">
          <td>#${v.rank}</td>
          <td><strong>V${v.id}</strong></td>
          <td><span class="sev-chip sev-${v.severity}">${v.severity.toUpperCase()}</span></td>
          <td>${v.distance !== null ? v.distance + ' steps' : 'Unreachable'}</td>
          <td>${v.risk !== null ? v.risk + ' pts' : '-'}</td>
          <td><code>${v.formula}</code></td>
          <td><span class="badge-pill ${v.rescued ? 'badge-green' : 'badge-amber'} victim-status">${v.rescued ? 'Planned' : 'Waiting'}</span></td>
        </tr>
      `).join("");
      document.getElementById("victimScoringRows").innerHTML = victimRows;
      document.getElementById("victimDecisionExplanation").innerHTML = `💡 <strong>AI Selection Reason:</strong> ${data.decision.reason}`;

      document.getElementById("knnVote").textContent = data.ml.knn;
      document.getElementById("nbVote").textContent = data.ml.naive_bayes;
      document.getElementById("dtVote").textContent = data.ml.decision_tree;
      document.getElementById("consensusVote").textContent = `${data.ml.final} PRIORITY`;

      document.getElementById("fzDist").textContent = `${data.fuzzy.distance} steps`;
      document.getElementById("fzRisk").textContent = `${data.fuzzy.risk} pts`;
      document.getElementById("fzProb").textContent = `${Math.round(data.fuzzy.blockage_probability * 100)}%`;
      document.getElementById("fzDecision").textContent = data.fuzzy.decision;
      document.getElementById("fuzzyImpactText").innerHTML = `💡 <strong>Uncertainty Reason:</strong> ${data.fuzzy.impact}`;

      const cspHtml = data.csp.ambulance_loads.map(load => {
        const victimsForAmb = data.assignments.filter(a => a.ambulance === load.ambulance);
        const victimChips = victimsForAmb.map(a => `
          <div class="amb-victim-chip">
            <span><strong>Victim V${a.victim}</strong> (${a.location})</span>
            <span class="sev-chip sev-${a.severity}">${a.severity}</span>
          </div>
        `).join("");

        return `
          <div class="ambulance-card">
            <div class="amb-head">
              <span>🚑 ${load.ambulance}</span>
              <span class="badge-pill ${load.count > 0 ? 'badge-green' : 'badge-blue'} ambulance-rescue-count" data-count="${load.count}">${load.count} planned</span>
            </div>
            <div class="amb-location">Current location: (${load.position.join(", ")})</div>
            <div class="capacity-meter">
              <div class="capacity-bar" style="width: ${Math.round(load.count / Math.max(1, load.capacity) * 100)}%;"></div>
            </div>
            <div class="amb-victims-list">
              ${victimChips || '<span style="font-size:10px; color:var(--text-dim);">Ready for dispatch</span>'}
            </div>
          </div>
        `;
      }).join("");
      document.getElementById("cspBoard").innerHTML = cspHtml;
      document.getElementById("cspExplanation").innerHTML = data.csp.explanation;

      document.getElementById("kpiSaved").textContent = data.kpis.victims_saved;
      document.getElementById("kpiLength").textContent = data.kpis.path_length;
      document.getElementById("kpiRisk").textContent = data.kpis.total_risk;
      document.getElementById("kpiUtil").textContent = data.kpis.fleet_util;
      document.getElementById("statVictims").textContent = data.victims.length;
      document.getElementById("statRescued").textContent = data.decision.saved;
      document.getElementById("statRescued").parentElement.querySelector("span").textContent = "Victims in rescue plan";
      document.getElementById("statAmbulances").textContent = data.ambulance_count;
      document.getElementById("statDistance").textContent = `${data.decision.path_length} steps`;

      if (data.dynamic_event) {
        eventTitle.innerHTML = data.dynamic_event.title;
        eventDescription.textContent = data.dynamic_event.description;
        eventTakeaway.innerHTML = `✓ <strong>AI Response:</strong> ${data.dynamic_event.ai_action} &mdash; <em>${data.dynamic_event.takeaway}</em>`;
        eventBanner.classList.add("active");
        log(`Dynamic Event Triggered: ${data.dynamic_event.title}`);
      } else {
        eventBanner.classList.remove("active");
      }

      updateAlgoButtons(data);
    }

    function updateAlgoButtons(data) {
      const container = document.getElementById("algoButtons");
      container.innerHTML = data.algorithms.map(algo => `
        <button class="btn-algo ${algo.name === selectedAlgo ? 'active' : ''}" data-algo="${algo.name}" type="button">
          ${algo.name} (${algo.length} steps, ${algo.risk} risk)
        </button>
      `).join("");

      attachAlgoClickHandlers();
      updateSelectedAlgoView();
      updateAlgorithmComparison(data);
    }

    function updateAlgorithmComparison(data) {
      const riskAware = data.algorithms.find(algo => algo.name === "Risk-Aware A*");
      const alternatives = data.algorithms
        .filter(algo => algo.name !== "Risk-Aware A*" && algo.reached)
        .sort((left, right) => left.weighted_cost - right.weighted_cost);
      const summary = document.getElementById("algoPerformanceSummary");

      if (!riskAware || !riskAware.reached) {
        summary.textContent = "Risk-Aware A* could not reach the selected victim in this scenario.";
      } else if (!alternatives.length) {
        summary.textContent = `Risk-Aware A* reached Victim V${data.decision.selected_victim} with weighted cost ${riskAware.weighted_cost}; no other algorithm found a complete route.`;
      } else {
        const bestAlternative = alternatives[0];
        const difference = bestAlternative.weighted_cost - riskAware.weighted_cost;
        const percent = bestAlternative.weighted_cost > 0
          ? (Math.abs(difference) / bestAlternative.weighted_cost * 100).toFixed(1)
          : null;
        let comparison;
        if (difference > 0) {
          comparison = `${percent}% lower weighted cost than ${bestAlternative.name}`;
        } else if (difference < 0) {
          comparison = `${percent || "0"}% higher weighted cost than ${bestAlternative.name}`;
        } else {
          comparison = `tied with ${bestAlternative.name} on weighted cost`;
        }
        summary.textContent = `Active rescue planner: Risk-Aware A*. For the outbound route to Victim V${data.decision.selected_victim}, it scored ${riskAware.weighted_cost} versus ${bestAlternative.weighted_cost} for the best alternative (${comparison}).`;
      }

      document.getElementById("algorithmComparisonRows").innerHTML = data.algorithms.map(algo => {
        let comparison = "No route";
        if (algo.reached && riskAware && riskAware.reached) {
          if (algo.name === "Risk-Aware A*") {
            comparison = "Reference";
          } else {
            const difference = algo.weighted_cost - riskAware.weighted_cost;
            const percent = riskAware.weighted_cost > 0
              ? (Math.abs(difference) / riskAware.weighted_cost * 100).toFixed(1)
              : null;
            comparison = difference > 0
              ? (percent === null ? `+${difference} cost` : `${percent}% higher`)
              : difference < 0
                ? (percent === null ? `${difference} cost` : `${percent}% lower`)
                : "Tied";
          }
        }

        return `
          <tr>
            <td><strong>${algo.name}</strong></td>
            <td>${algo.reached ? algo.length : "—"}</td>
            <td>${algo.reached ? algo.risk : "—"}</td>
            <td>${algo.reached ? algo.weighted_cost : "—"}</td>
            <td>${comparison}</td>
          </tr>
        `;
      }).join("");
    }

    function attachAlgoClickHandlers() {
      document.querySelectorAll(".btn-algo").forEach(btn => {
        btn.addEventListener("click", () => {
          selectedAlgo = btn.dataset.algo;
          document.querySelectorAll(".btn-algo").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          updateSelectedAlgoView();
        });
      });
    }

    function updateSelectedAlgoView() {
      const algoObj = currentData.algorithms.find(a => a.name === selectedAlgo);
      if (!algoObj) return;

      document.getElementById("activeAlgoBadge").textContent = `${algoObj.name} (${algoObj.badge})`;
      document.getElementById("algoExplanation").innerHTML = `
        <strong>${algoObj.name} [${algoObj.optimality} &bull; ${algoObj.safety}]:</strong> ${algoObj.explanation}
      `;
      renderGrid(currentData, algoObj.path || []);
    }

    // Animation runner
    function stopRouteAnimation(clear = false) {
      if (animationTimer) {
        clearTimeout(animationTimer);
        animationTimer = null;
      }
      isPaused = false;
      pauseBtn.textContent = "⏸ Pause";
      if (clear) {
        routeCoords = [];
        routeStepIndex = 0;
        rescuedVictimIds = new Set();
        document.getElementById("rescueNotifications").textContent = "";
        renderGrid(currentData, currentData.paths.routes.flat(), null, rescuedVictimIds);
      }
    }

    function updateVictimStatus(victimId, statusText) {
      const row = document.querySelector(`#victimScoringRows tr[data-victim-id="${victimId}"]`);
      if (!row) return;
      const status = row.querySelector(".victim-status");
      if (!status) return;
      status.textContent = statusText;
      status.className = "badge-pill badge-green victim-status";
    }

    function stepRouteAnimation() {
      if (isPaused) return;

      if (routeStepIndex < routeCoords.length) {
        const step = routeCoords[routeStepIndex];
        const amb = step.position;
        const activeTrail = routeCoords.slice(0, routeStepIndex + 1).map(item => item.position);
        if (step.victimId !== null) {
          rescuedVictimIds.add(step.victimId);
          updateVictimStatus(step.victimId, "Picked up");
          notifyVictimRescued(step.victimId);
          log(`Victim V${step.victimId} picked up; marker removed from the map.`);
        }
        if (step.deliveredVictimId !== null) {
          updateVictimStatus(step.deliveredVictimId, "Delivered");
        }
        renderGrid(currentData, activeTrail, amb, rescuedVictimIds);
        routeStepIndex++;
        animationTimer = setTimeout(stepRouteAnimation, 220);
      } else {
        animationTimer = null;
        document.querySelectorAll(".victim-status").forEach(status => {
          if (status.textContent.trim() === "Planned") status.textContent = "Delivered";
        });
        document.querySelectorAll(".ambulance-rescue-count").forEach(status => {
          status.textContent = `${status.dataset.count} delivered`;
        });
        document.querySelector("#statRescued").parentElement.querySelector("span").textContent = "Victims delivered";
        log(`Rescue plan complete: ${currentData.decision.saved} of ${currentData.victims.length} victims delivered to hospital.`);
      }
    }

    function startRouteAnimation() {
      stopRouteAnimation(true);
      currentData.dispatches.forEach((dispatch, dispatchIndex) => {
        let victimReached = false;
        (dispatch.path || []).forEach((position, pathIndex) => {
          if (dispatchIndex > 0 && pathIndex === 0) return;
          const atVictim = position[0] === dispatch.location[0] && position[1] === dispatch.location[1];
          if (atVictim) victimReached = true;
          const atHospital = victimReached && position[0] === dispatch.hospital[0] && position[1] === dispatch.hospital[1];
          routeCoords.push({
            position,
            victimId: atVictim ? dispatch.victim : null,
            deliveredVictimId: atHospital ? dispatch.victim : null,
          });
        });
      });
      routeStepIndex = 0;
      currentData.dispatches.forEach(dispatch => {
        log(`${dispatch.ambulance} dispatched from (${dispatch.from.join(",")}) for Victim V${dispatch.victim}; returning to Hospital ${dispatch.hospital_label} (${dispatch.hospital.join(",")}).`);
      });
      stepRouteAnimation();
    }

    // API fetch helper
    async function fetchDashboard(params = {}, bodyPayload = null) {
      const url = new URL("/api/dashboard", window.location.origin);
      Object.keys(params).forEach(k => url.searchParams.append(k, params[k]));
      url.searchParams.set("ambulance_count", ambulanceCountSelect.value);

      loadingOverlay.classList.add("active");
      try {
        const options = bodyPayload ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(bodyPayload)
        } : { method: "GET" };

        const res = await fetch(url, options);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Dashboard request failed");
        currentData = data;
        ambulanceCountSelect.value = data.ambulance_count;
        studioScenario = JSON.parse(JSON.stringify(data.scenario));
        rescuedVictimIds = new Set();
        renderGrid(data, data.paths.routes.flat(), null, rescuedVictimIds);
        updatePanels(data);
        return data;
      } finally {
        setTimeout(() => loadingOverlay.classList.remove("active"), 300);
      }
    }

    // ==========================================
    // SCENARIO STUDIO & VISUAL PAINTER ENGINE
    // ==========================================
    const painterGrid = document.getElementById("painterGrid");
    const studioSizeSelect = document.getElementById("studioSizeSelect");
    const studioStatsTally = document.getElementById("studioStatsTally");

    // Tool selection
    document.querySelectorAll(".btn-tool").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-tool").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activePaintTool = btn.dataset.tool;
      });
    });

    function syncStudioUi() {
      studioSizeSelect.value = studioScenario.size;
      renderPainterGrid();
      updateStudioTally();
      document.getElementById("jsonScenarioEditor").value = JSON.stringify(studioScenario, null, 2);
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function updateStudioTally() {
      const crit = studioScenario.victims.filter(v => v.severity === "critical").length;
      const mod = studioScenario.victims.filter(v => v.severity === "moderate").length;
      const min = studioScenario.victims.filter(v => v.severity === "minor").length;
      const hospitalLocations = studioScenario.hospitals.map((hospital, index) =>
        `H${index + 1} (${escapeHtml(hospital[0])},${escapeHtml(hospital[1])})`
      ).join(", ");
      const victimLocations = studioScenario.victims.map(victim =>
        `V${escapeHtml(victim.id)} (${escapeHtml(victim.x)},${escapeHtml(victim.y)}) ${escapeHtml(victim.severity)}`
      ).join("; ");
      studioStatsTally.innerHTML = `
        Base <strong>BASE (${escapeHtml(studioScenario.base[0])},${escapeHtml(studioScenario.base[1])})</strong> &bull;
        Hospitals <strong>${hospitalLocations || "none"}</strong> &bull;
        Victims <strong>${victimLocations || "none"}</strong> (Crit:${crit}, Mod:${mod}, Min:${min}) &bull;
        Roadblocks <strong>${studioScenario.blocked_cells.length}</strong> &bull;
        Fire hazards <strong>${studioScenario.risk_zones.length}</strong>
      `;
    }

    function renderPainterGrid() {
      const size = studioScenario.size;
      const cellPx = Math.max(24, Math.min(38, Math.floor(400 / size)));
      painterGrid.style.gridTemplateColumns = `repeat(${size}, ${cellPx}px)`;

      // Maps
      const baseKey = `${studioScenario.base[0]},${studioScenario.base[1]}`;
      const hospitalMap = new Map(studioScenario.hospitals.map((h, index) => [`${h[0]},${h[1]}`, `H${index + 1}`]));
      const victimMap = new Map(studioScenario.victims.map(v => [`${v.x},${v.y}`, v]));
      const riskMap = new Map(studioScenario.risk_zones.map(r => [`${r.x},${r.y}`, r.level]));
      const blockSet = new Set(studioScenario.blocked_cells.map(b => `${b[0]},${b[1]}`));

      let html = "";
      for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
          const key = `${x},${y}`;
          let cellClass = "cell";
          let label = "";
          let cellTitle = "Open road";

          if (key === baseKey) {
            cellClass += " base"; label = "BASE"; cellTitle = "Base";
          } else if (hospitalMap.has(key)) {
            const hospitalLabel = hospitalMap.get(key);
            cellClass += " hospital"; label = hospitalLabel; cellTitle = `Hospital ${hospitalLabel}`;
          } else if (victimMap.has(key)) {
            cellClass += " victim";
            const v = victimMap.get(key);
            const severity = ["critical", "moderate", "minor"].includes(v.severity) ? v.severity : "critical";
            cellClass += ` victim-${severity}`;
            label = `V${escapeHtml(v.id)}`;
            cellTitle = `Victim V${escapeHtml(v.id)} (${escapeHtml(severity)})`;
          } else if (blockSet.has(key)) {
            cellClass += " blocked"; label = "BLOCK"; cellTitle = "Blocked road";
          } else if (riskMap.has(key)) {
            const rawLevel = Number(riskMap.get(key));
            const lvl = [1, 2, 3].includes(rawLevel) ? rawLevel : 1;
            cellClass += ` risk risk-${lvl}`; label = `L${lvl}`; cellTitle = `Fire hazard Level ${lvl}`;
          }

          html += `
            <div class="${cellClass}" style="width:${cellPx}px; height:${cellPx}px;"
                 data-x="${x}" data-y="${y}" title="${escapeHtml(cellTitle)} (${x}, ${y})">
              ${escapeHtml(label)}
            </div>
          `;
        }
      }
      painterGrid.innerHTML = html;

      // Click cell to paint
      painterGrid.querySelectorAll(".cell").forEach(c => {
        c.addEventListener("click", () => {
          const x = parseInt(c.dataset.x, 10);
          const y = parseInt(c.dataset.y, 10);
          applyPaintTool(x, y);
        });
      });
    }

    function applyPaintTool(x, y) {
      // Clear anything on this cell first if modifying
      const isBase = (studioScenario.base[0] === x && studioScenario.base[1] === y);
      if (activePaintTool === "base") {
        studioScenario.base = [x, y];
        // Remove conflicts
        studioScenario.hospitals = studioScenario.hospitals.filter(h => !(h[0] === x && h[1] === y));
        studioScenario.victims = studioScenario.victims.filter(v => !(v.x === x && v.y === y));
        studioScenario.blocked_cells = studioScenario.blocked_cells.filter(b => !(b[0] === x && b[1] === y));
        studioScenario.risk_zones = studioScenario.risk_zones.filter(r => !(r.x === x && r.y === y));
      } else if (isBase) {
        return; // Don't overwrite base with other tools unless moving base
      } else if (activePaintTool === "erase") {
        studioScenario.hospitals = studioScenario.hospitals.filter(h => !(h[0] === x && h[1] === y));
        studioScenario.victims = studioScenario.victims.filter(v => !(v.x === x && v.y === y));
        studioScenario.blocked_cells = studioScenario.blocked_cells.filter(b => !(b[0] === x && b[1] === y));
        studioScenario.risk_zones = studioScenario.risk_zones.filter(r => !(r.x === x && r.y === y));
      } else if (activePaintTool === "hospital") {
        const idx = studioScenario.hospitals.findIndex(h => h[0] === x && h[1] === y);
        if (idx >= 0) {
          studioScenario.hospitals.splice(idx, 1);
        } else {
          // Clear any victim/block/risk
          studioScenario.victims = studioScenario.victims.filter(v => !(v.x === x && v.y === y));
          studioScenario.blocked_cells = studioScenario.blocked_cells.filter(b => !(b[0] === x && b[1] === y));
          studioScenario.risk_zones = studioScenario.risk_zones.filter(r => !(r.x === x && r.y === y));
          studioScenario.hospitals.push([x, y]);
        }
      } else if (activePaintTool.startsWith("victim_")) {
        const sev = activePaintTool.replace("victim_", "");
        studioScenario.hospitals = studioScenario.hospitals.filter(h => !(h[0] === x && h[1] === y));
        studioScenario.blocked_cells = studioScenario.blocked_cells.filter(b => !(b[0] === x && b[1] === y));
        studioScenario.risk_zones = studioScenario.risk_zones.filter(r => !(r.x === x && r.y === y));

        const existing = studioScenario.victims.find(v => v.x === x && v.y === y);
        if (existing) {
          existing.severity = sev;
        } else {
          const nextId = (studioScenario.victims.length ? Math.max(...studioScenario.victims.map(v => v.id)) : 0) + 1;
          studioScenario.victims.push({ id: nextId, x, y, severity: sev });
        }
      } else if (activePaintTool === "block") {
        const idx = studioScenario.blocked_cells.findIndex(b => b[0] === x && b[1] === y);
        if (idx >= 0) {
          studioScenario.blocked_cells.splice(idx, 1);
        } else {
          studioScenario.hospitals = studioScenario.hospitals.filter(h => !(h[0] === x && h[1] === y));
          studioScenario.victims = studioScenario.victims.filter(v => !(v.x === x && v.y === y));
          studioScenario.risk_zones = studioScenario.risk_zones.filter(r => !(r.x === x && r.y === y));
          studioScenario.blocked_cells.push([x, y]);
        }
      } else if (activePaintTool.startsWith("risk_")) {
        const lvl = parseInt(activePaintTool.replace("risk_", ""), 10);
        studioScenario.hospitals = studioScenario.hospitals.filter(h => !(h[0] === x && h[1] === y));
        studioScenario.victims = studioScenario.victims.filter(v => !(v.x === x && v.y === y));
        studioScenario.blocked_cells = studioScenario.blocked_cells.filter(b => !(b[0] === x && b[1] === y));

        const existing = studioScenario.risk_zones.find(r => r.x === x && r.y === y);
        if (existing) {
          existing.level = lvl;
        } else {
          studioScenario.risk_zones.push({ x, y, level: lvl });
        }
      }

      renderPainterGrid();
      updateStudioTally();
      document.getElementById("jsonScenarioEditor").value = JSON.stringify(studioScenario, null, 2);
    }

    // Change size in studio
    studioSizeSelect.addEventListener("change", () => {
      const newSize = parseInt(studioSizeSelect.value, 10);
      studioScenario.size = newSize;
      // Prune coordinates outside size
      if (studioScenario.base[0] >= newSize || studioScenario.base[1] >= newSize) studioScenario.base = [0, 0];
      studioScenario.hospitals = studioScenario.hospitals.filter(h => h[0] < newSize && h[1] < newSize);
      if (!studioScenario.hospitals.length) studioScenario.hospitals = [[newSize - 1, newSize - 1]];
      studioScenario.victims = studioScenario.victims.filter(v => v.x < newSize && v.y < newSize);
      studioScenario.risk_zones = studioScenario.risk_zones.filter(r => r.x < newSize && r.y < newSize);
      studioScenario.blocked_cells = studioScenario.blocked_cells.filter(b => b[0] < newSize && b[1] < newSize);

      renderPainterGrid();
      updateStudioTally();
      document.getElementById("jsonScenarioEditor").value = JSON.stringify(studioScenario, null, 2);
    });

    document.getElementById("studioClearObstaclesBtn").addEventListener("click", () => {
      studioScenario.blocked_cells = [];
      studioScenario.risk_zones = [];
      renderPainterGrid();
      updateStudioTally();
      document.getElementById("jsonScenarioEditor").value = JSON.stringify(studioScenario, null, 2);
    });

    document.getElementById("studioResetBtn").addEventListener("click", () => {
      studioScenario = JSON.parse(JSON.stringify(currentData.scenario));
      syncStudioUi();
    });

    // Apply Scenario to Live Simulation
    document.getElementById("studioApplyBtn").addEventListener("click", async () => {
      try {
        if (!studioScenario.hospitals.length) {
          alert("Please place at least one hospital on the map.");
          return;
        }
        if (!studioScenario.victims.length) {
          alert("Please place at least one casualty (victim) on the map.");
          return;
        }
        stopRouteAnimation(true);
        configModal.classList.remove("active");
        await fetchDashboard({}, studioScenario);
        log(`Applied custom scenario (${studioScenario.size}x${studioScenario.size}, ${studioScenario.victims.length} casualties). Recomputed route.`);
      } catch (err) {
        alert("Error applying scenario: " + err.message);
      }
    });

    // Preset Loading
    document.querySelectorAll(".btn-load-preset").forEach(btn => {
      btn.addEventListener("click", async () => {
        const presetKey = btn.dataset.preset;
        stopRouteAnimation(true);
        configModal.classList.remove("active");
        presetSelect.value = presetKey;
        await fetchDashboard({ preset: presetKey });
        log(`Loaded preset template: ${presetKey.toUpperCase()}`);
      });
    });

    // Procedural Generator in Studio
    document.getElementById("studioGenerateBtn").addEventListener("click", async () => {
      const size = parseInt(document.getElementById("genSize").value, 10);
      const victims = parseInt(document.getElementById("genVictims").value, 10);
      const hospitals = parseInt(document.getElementById("genHospitals").value, 10);
      const hazards = document.getElementById("genHazards").value;
      const blocks = document.getElementById("genBlocks").value;

      loadingOverlay.classList.add("active");
      try {
        const res = await fetch(`/api/scenario/generate?size=${size}&victims=${victims}&hospitals=${hospitals}&hazards=${hazards}&blocks=${blocks}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Generation failed");
        studioScenario = data.scenario;
        syncStudioUi();
        // Switch to Painter tab
        document.querySelectorAll("#studioTabs .guide-tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll("#configModal .guide-tab-pane").forEach(p => p.classList.remove("active"));
        document.querySelector("[data-tab='painter']").classList.add("active");
        document.getElementById("tab-painter").classList.add("active");
      } catch (err) {
        alert("Generation error: " + err.message);
      } finally {
        loadingOverlay.classList.remove("active");
      }
    });

    // Quick Roll Random button on main bar
    quickRollRandomBtn.addEventListener("click", async () => {
      stopRouteAnimation(true);
      loadingOverlay.classList.add("active");
      try {
        const res = await fetch("/api/scenario/generate?size=10&victims=4&hospitals=2&hazards=medium&blocks=medium");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        await fetchDashboard({}, data.scenario);
        log("Procedural random disaster map generated and activated.");
      } catch (err) {
        alert(err.message);
      } finally {
        loadingOverlay.classList.remove("active");
      }
    });

    // JSON Studio Handlers
    document.getElementById("copyJsonBtn").addEventListener("click", () => {
      navigator.clipboard.writeText(document.getElementById("jsonScenarioEditor").value);
      alert("Scenario JSON copied to clipboard!");
    });

    document.getElementById("formatJsonBtn").addEventListener("click", () => {
      try {
        const parsed = JSON.parse(document.getElementById("jsonScenarioEditor").value);
        document.getElementById("jsonScenarioEditor").value = JSON.stringify(parsed, null, 2);
        document.getElementById("jsonErrorStatus").textContent = "✓ Valid JSON format.";
      } catch (e) {
        document.getElementById("jsonErrorStatus").textContent = "Invalid JSON: " + e.message;
      }
    });

    document.getElementById("studioImportJsonBtn").addEventListener("click", () => {
      try {
        const parsed = JSON.parse(document.getElementById("jsonScenarioEditor").value);
        if (!parsed.size || !parsed.base || !parsed.hospitals || !parsed.victims) {
          throw new Error("Missing required fields: size, base, hospitals, victims");
        }
        studioScenario = parsed;
        syncStudioUi();
        // Switch to Painter tab
        document.querySelectorAll("#studioTabs .guide-tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll("#configModal .guide-tab-pane").forEach(p => p.classList.remove("active"));
        document.querySelector("[data-tab='painter']").classList.add("active");
        document.getElementById("tab-painter").classList.add("active");
        document.getElementById("jsonErrorStatus").textContent = "Loaded onto canvas successfully!";
      } catch (e) {
        document.getElementById("jsonErrorStatus").textContent = "JSON Error: " + e.message;
      }
    });

    // Main Control Listeners
    startBtn.addEventListener("click", async () => {
      await fetchDashboard();
      startRouteAnimation();
    });

    pauseBtn.addEventListener("click", () => {
      if (!routeCoords.length) return;
      isPaused = !isPaused;
      pauseBtn.textContent = isPaused ? "▶ Resume" : "⏸ Pause";
      log(isPaused ? "Mission animation paused." : "Mission animation resumed.");
      if (!isPaused) stepRouteAnimation();
    });

    resetBtn.addEventListener("click", async () => {
      stopRouteAnimation(true);
      await fetchDashboard({ reset: "1" });
      log("Reset scenario to default grid.");
    });

    ambulanceCountSelect.addEventListener("change", async () => {
      stopRouteAnimation(true);
      await fetchDashboard();
      log(`Fleet updated: ${currentData.ambulance_count} ambulance(s) available. Rescue plan recalculated.`);
    });

    presetSelect.addEventListener("change", async () => {
      stopRouteAnimation(true);
      const preset = presetSelect.value;
      await fetchDashboard({ preset });
      log(`Loaded preset scenario: ${preset.toUpperCase()}`);
    });

    // Dynamic Event Handlers
    document.getElementById("evBlockBtn").addEventListener("click", async () => {
      stopRouteAnimation(true);
      await fetchDashboard({ event: "blockage" });
      startRouteAnimation();
    });

    document.getElementById("evRiskBtn").addEventListener("click", async () => {
      stopRouteAnimation(true);
      await fetchDashboard({ event: "risk" });
      startRouteAnimation();
    });

    document.getElementById("evVictimBtn").addEventListener("click", async () => {
      stopRouteAnimation(true);
      await fetchDashboard({ event: "new_victim" });
      startRouteAnimation();
    });

    document.getElementById("evCspBtn").addEventListener("click", async () => {
      stopRouteAnimation(true);
      await fetchDashboard({ event: "ambulance_loss" });
      startRouteAnimation();
    });

    // Initial render
    renderGrid(initialData, initialData.paths.routes.flat());
    attachAlgoClickHandlers();
    updateAlgorithmComparison(initialData);
  </script>
</body>
</html>
"""


TAB_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} — AIDRA Mission Control</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f3f6fc;
      --bg-alt: #f7f9fe;
      --panel: #ffffff;
      --panel-card: #f3f6fc;
      --border: #dce5f4;
      --border-light: #c6d5ec;
      --text: #172342;
      --text-muted: #52617f;
      --blue: #1268e8;
      --emerald: #10b981;
      --amber: #f59e0b;
      --purple: #8b5cf6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      min-height: 100vh;
      line-height: 1.5;
    }
    a, button { color: inherit; text-decoration: none; font-family: inherit; }
    .top-header {
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 10px 20px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand-wrap { display: flex; align-items: center; gap: 12px; }
    .brand-logo { width: 36px; height: 36px; border-radius: 9px; background: linear-gradient(135deg, #2563eb, #7c3aed); display: flex; align-items: center; justify-content: center; font-size: 18px; }
    .nav-tabs { display: flex; flex-wrap: wrap; gap: 4px; background: #f5f8fe; padding: 4px; border-radius: 12px; border: 1px solid var(--border); }
    .nav-tab { padding: 7px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; color: var(--text-muted); white-space: nowrap; }
    .nav-tab.active { background: var(--blue); color: #fff; }
    .page-container { max-width: 1400px; margin: 0 auto; padding: 24px 20px 50px; display: flex; flex-direction: column; gap: 20px; }
    .hero-banner { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 20px 24px; }
    .hero-banner h2 { font-size: 20px; font-weight: 800; color: var(--text); margin-bottom: 4px; }
    .hero-banner p { color: var(--text-muted); font-size: 13px; }
    .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 18px; display: flex; flex-direction: column; gap: 14px; }
    .panel h3 { font-size: 13px; font-weight: 800; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { background: var(--panel-card); color: var(--text-muted); font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
    td { padding: 9px 10px; border-bottom: 1px solid rgba(23, 35, 66, 0.08); color: var(--text-muted); }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .pill-green { background: rgba(16, 185, 129, 0.1); color: #047857; }
    .pill-amber { background: rgba(245, 158, 11, 0.12); color: #a16207; }
    .pill-red { background: rgba(239, 68, 68, 0.1); color: #b91c1c; }
    .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(23, 35, 66, 0.08); font-size: 12.5px; }
    .metric-row:last-child { border-bottom: 0; }
    .metric-row span { color: var(--text-muted); }
    .metric-row strong { color: var(--text); font-weight: 700; }
    .code-trace { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px; padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--text-muted); line-height: 1.6; white-space: pre-wrap; max-height: 320px; overflow-y: auto; }
    .chart-container { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 18px; }
    .chart-container h3 { font-size: 13px; font-weight: 800; color: #1d4ed8; margin-bottom: 12px; }
    .chart-container canvas { height: 260px !important; width: 100% !important; }
    @media (max-width: 720px) {
      .top-header { padding: 10px; }
      .nav-tabs { width: 100%; overflow-x: auto; flex-wrap: nowrap; }
      .page-container { padding: 16px 10px 32px; }
      .hero-banner { padding: 16px; }
    }
  </style>
</head>
<body>
  <header class="top-header">
    <div class="brand-wrap">
      <div class="brand-logo">🚑</div>
      <div><strong>AIDRA Mission Control</strong></div>
    </div>
    <nav class="nav-tabs">
      <a class="nav-tab" href="/">Overview</a>
      <a class="nav-tab {{ 'active' if active == 'search' else '' }}" href="/search-trace">Route algorithms</a>
      <a class="nav-tab {{ 'active' if active == 'csp' else '' }}" href="/csp-solver">Fleet planning</a>
      <a class="nav-tab {{ 'active' if active == 'ml' else '' }}" href="/ml-studio">ML Studio</a>
      <a class="nav-tab {{ 'active' if active == 'analytics' else '' }}" href="/analytics">Analytics</a>
      <a class="nav-tab" href="/simulation">Simulation log</a>
    </nav>
  </header>

  <main class="page-container">
    <div class="hero-banner">
      <h2>{{ title }}</h2>
      <p>{{ subtitle }}</p>
    </div>

    {% if active == "search" %}
      <div class="card-grid">
        <div class="panel">
          <h3>Target Selection Overview</h3>
          <div class="metric-row"><span>Target victim</span><strong>Victim V{{ data.decision.selected_victim }}</strong></div>
          <div class="metric-row"><span>Victim severity</span><strong>{{ data.decision.selected_severity|upper }}</strong></div>
          <div class="metric-row"><span>Active Algorithm</span><strong>Risk-Aware A*</strong></div>
          <div class="metric-row"><span>Destination hospital</span><strong>{{ data.decision.selected_hospital_label }} ({{ data.decision.selected_hospital[0] }}, {{ data.decision.selected_hospital[1] }})</strong></div>
          <div class="metric-row"><span>Total Rescue Steps</span><strong>{{ data.decision.path_length }} steps</strong></div>
          <div class="metric-row"><span>Hazard Risk Incurred</span><strong>{{ data.decision.total_risk }} pts</strong></div>
        </div>

        <div class="panel">
          <h3>Algorithm Comparison Matrix</h3>
          <p>Risk-Aware A* is the active route planner. Weighted cost = steps + {{ data.route_risk_weight }} × risk points; lower is better.</p>
          <table>
            <thead>
              <tr><th>Algorithm</th><th>Steps</th><th>Risk</th><th>Weighted Cost</th><th>Optimality</th><th>Status</th></tr>
            </thead>
            <tbody>
              {% for algo in data.algorithms %}
                <tr>
                  <td><strong>{{ algo.name }}</strong></td>
                  <td>{{ algo.length if algo.reached else "—" }}</td>
                  <td>{{ algo.risk if algo.reached else "—" }}</td>
                  <td>{{ algo.weighted_cost if algo.reached else "—" }}</td>
                  <td>{{ algo.optimality }}</td>
                  <td><span class="pill {{ 'pill-green' if algo.reached else 'pill-amber' }}">{{ "Reached" if algo.reached else "Trapped" }}</span></td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <h3>Step-by-Step Trajectory Traces</h3>
        <div class="code-trace">Base to Victim V{{ data.decision.selected_victim }} path:
{{ data.paths.base_to_victim }}

Victim to hospital path:
{{ data.paths.victim_to_hospital }}

Combined Complete Mission Trajectory:
{{ data.paths.full }}</div>
      </div>

    {% elif active == "csp" %}
      <div class="card-grid">
        <div class="panel">
          <h3>Ambulance Allocation Matrix</h3>
          <table>
            <thead><tr><th>Victim</th><th>Severity</th><th>Coordinates</th><th>Assigned unit</th><th>Status</th></tr></thead>
            <tbody>
              {% for item in data.assignments %}
                <tr>
                  <td><strong>Victim V{{ item.victim }}</strong></td>
                  <td><span class="pill {{ 'pill-green' if item.severity=='minor' else ('pill-amber' if item.severity=='moderate' else 'pill-red') }}">{{ item.severity|upper }}</span></td>
                  <td>{{ item.location }}</td>
                  <td><strong>{{ item.ambulance }}</strong></td>
                  <td><span class="pill {{ 'pill-green' if item.assigned else 'pill-amber' }}">{{ "Assigned" if item.assigned else "Queued" }}</span></td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>

        <div class="panel">
          <h3>CSP Constraint Verification</h3>
          <div class="metric-row"><span>Fleet capacity limit</span><strong>Max 2 victims per ambulance</strong></div>
          <div class="metric-row"><span>Victims assigned</span><strong>{{ data.csp.assigned_count }} / {{ data.csp.total_victims }}</strong></div>
          <div class="metric-row"><span>Backtracking Steps</span><strong>{{ data.csp.backtrack_count }} steps</strong></div>
          <div class="metric-row"><span>Heuristic Method</span><strong>MRV (Minimum Remaining Values)</strong></div>
          <div class="metric-row"><span>Constraint State</span><strong>{{ "100% Feasible" if data.csp.all_assigned else "Capacity Bottleneck (Wave 2 Queued)" }}</strong></div>
          {% for load in data.csp.ambulance_loads %}
            <div class="metric-row"><span>{{ load.ambulance }} load</span><strong>{{ load.count }} / {{ load.capacity }} victims</strong></div>
          {% endfor %}
        </div>
      </div>

      <div class="panel">
        <h3>Backtracking Search Tree Trace</h3>
        <div class="code-trace">{{ data.csp.trace_text if data.csp.trace_text else "No backtrack exploration required." }}</div>
      </div>

    {% elif active == "ml" %}
      <div class="card-grid">
        <div class="panel">
          <h3>Ensemble Classification Vote</h3>
          <div class="metric-row"><span>k-Nearest Neighbors (kNN)</span><strong>{{ data.ml.knn }}</strong></div>
          <div class="metric-row"><span>Gaussian Naive Bayes</span><strong>{{ data.ml.naive_bayes }}</strong></div>
          <div class="metric-row"><span>Decision Tree</span><strong>{{ data.ml.decision_tree }}</strong></div>
          <div class="metric-row"><span>Majority Consensus</span><strong style="color:#6d28d9;">{{ data.ml.final }} PRIORITY</strong></div>
        </div>

        <div class="panel">
          <h3>Fuzzy Logic Uncertainty Reasoning</h3>
          <div class="metric-row"><span>Distance Input</span><strong>{{ data.fuzzy.distance }} steps</strong></div>
          <div class="metric-row"><span>Risk Input</span><strong>{{ data.fuzzy.risk }} hazard pts</strong></div>
          <div class="metric-row"><span>Road Blockage Probability</span><strong>{{ (data.fuzzy.blockage_probability * 100)|int }}%</strong></div>
          <div class="metric-row"><span>Fuzzy Urgency Output</span><strong style="color:#a16207;">{{ data.fuzzy.decision }}</strong></div>
        </div>
      </div>

      <div class="panel">
        <h3>FEMA Dataset Supervised Model Performance</h3>
        <table>
          <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
          <tbody>
            <tr><td><strong>k-NN</strong></td><td>{{ data.ml.metrics.knn.accuracy }}</td><td>{{ data.ml.metrics.knn.precision }}</td><td>{{ data.ml.metrics.knn.recall }}</td><td>{{ data.ml.metrics.knn.f1 }}</td></tr>
            <tr><td><strong>Naive Bayes</strong></td><td>{{ data.ml.metrics.naive_bayes.accuracy }}</td><td>{{ data.ml.metrics.naive_bayes.precision }}</td><td>{{ data.ml.metrics.naive_bayes.recall }}</td><td>{{ data.ml.metrics.naive_bayes.f1 }}</td></tr>
            <tr><td><strong>Decision Tree</strong></td><td>{{ data.ml.metrics.decision_tree.accuracy }}</td><td>{{ data.ml.metrics.decision_tree.precision }}</td><td>{{ data.ml.metrics.decision_tree.recall }}</td><td>{{ data.ml.metrics.decision_tree.f1 }}</td></tr>
          </tbody>
        </table>
      </div>

    {% elif active == "analytics" %}
      <div class="card-grid">
        <div class="chart-container">
          <h3>Route Length by Algorithm (Fewer Steps)</h3>
          <canvas id="lengthChart"></canvas>
        </div>
        <div class="chart-container">
          <h3>Hazard Risk Exposure (Lower is Safer)</h3>
          <canvas id="riskChart"></canvas>
        </div>
      </div>

      <div class="panel">
        <h3>Algorithm Trade-off Analysis</h3>
        <table>
          <thead><tr><th>Algorithm</th><th>Speed</th><th>Optimality</th><th>Hazard Safety</th><th>Recommended Use Case</th></tr></thead>
          <tbody>
            {% for algo in data.analytics.comparison %}
              <tr>
                <td><strong>{{ algo.name }}</strong></td>
                <td>{{ algo.speed }}</td>
                <td>{{ algo.optimality }}</td>
                <td>{{ algo.safety }}</td>
                <td>{{ algo.use_case }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    {% endif %}
  </main>

  {% if active == "analytics" %}
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const rows = {{ data.analytics.comparison|tojson }};
    const labels = rows.map(r => r.name);
    const lengths = rows.map(r => r.length);
    const risks = rows.map(r => r.risk);

    const colors = ["#3b82f6", "#10b981", "#6366f1", "#ef4444", "#f59e0b", "#8b5cf6"];

    new Chart(document.getElementById("lengthChart"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Step Count", data: lengths, backgroundColor: colors }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
    });

    new Chart(document.getElementById("riskChart"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Risk Exposure", data: risks, backgroundColor: colors }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
    });
  </script>
  {% endif %}
</body>
</html>
"""


SIMULATION_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simulation log — AIDRA</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root { --bg: #f3f6fc; --bg-alt: #f7f9fe; --panel: #fff; --border: #dce5f4; --text: #172342; --text-muted: #52617f; --blue: #1268e8; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: 'Plus Jakarta Sans', sans-serif; min-height: 100vh; }
    .top-header { position: sticky; top: 0; z-index: 100; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding: 10px 20px; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--border); }
    .brand-wrap { display: flex; align-items: center; gap: 10px; font-weight: 800; }
    .brand-logo { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 9px; background: linear-gradient(135deg, #2563eb, #7c3aed); }
    .nav-tabs { display: flex; flex-wrap: wrap; gap: 4px; background: #f5f8fe; padding: 4px; border: 1px solid var(--border); border-radius: 12px; }
    .nav-tab { padding: 7px 12px; border-radius: 8px; color: var(--text-muted); font-size: 12px; font-weight: 600; text-decoration: none; white-space: nowrap; }
    .nav-tab.active { background: var(--blue); color: #fff; }
    .page-container { max-width: 1240px; margin: 0 auto; padding: 24px 20px 50px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }
    .panel-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
    h2 { margin: 0; color: var(--text); font-size: 20px; }
    .back-btn { background: #2563eb; color: #fff; padding: 8px 14px; border-radius: 8px; text-decoration: none; font-size: 12px; font-weight: 700; }
    pre { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 12px; padding: 18px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-muted); overflow: auto; line-height: 1.6; white-space: pre-wrap; }
  </style>
</head>
<body>
  <header class="top-header">
    <div class="brand-wrap"><span class="brand-logo">🚑</span><span>AIDRA Mission Control</span></div>
    <nav class="nav-tabs">
      <a class="nav-tab" href="/">Overview</a>
      <a class="nav-tab" href="/search-trace">Route algorithms</a>
      <a class="nav-tab" href="/csp-solver">Fleet planning</a>
      <a class="nav-tab" href="/ml-studio">ML Studio</a>
      <a class="nav-tab" href="/analytics">Analytics</a>
      <a class="nav-tab active" href="/simulation">Simulation log</a>
    </nav>
  </header>
  <main class="page-container">
    <div class="panel">
      <div class="panel-head">
        <h2>Simulation log</h2>
        <a class="back-btn" href="/">Back to overview</a>
      </div>
      <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">
        Status: <strong>{{ "Success" if result.ok else "Failed" }}</strong> | Exit Code: <strong>{{ result.exit_code }}</strong> | Duration: <strong>{{ result.duration_seconds }}s</strong>
      </div>
      <pre>{{ result.output }}</pre>
    </div>
  </main>
</body>
</html>
"""


def render_dashboard(scenario=None, error=None, status_code=200, event=None, preset=None, ambulance_count=2):
    scenario = scenario or DEFAULT_SCENARIO
    try:
        form = scenario_form_data(scenario)
        data = build_dashboard_data(scenario, event=event, preset=preset, ambulance_count=ambulance_count)
        if status_code == 200:
            set_active_scenario(data["scenario"])
    except ValueError as exc:
        clear_active_scenario()
        form = scenario_form_data(DEFAULT_SCENARIO)
        data = build_dashboard_data(DEFAULT_SCENARIO, ambulance_count=2)
        error = str(exc)
        status_code = 400

    return render_template_string(
        INDEX_HTML,
        data=data,
        form=form,
        error=error,
    ), status_code


def render_tab_page(title, subtitle, active):
    return render_template_string(
        TAB_PAGE_HTML,
        title=title,
        subtitle=subtitle,
        active=active,
        data=build_dashboard_data(get_active_scenario()),
    )


@app.get("/")
def index():
    try:
        ambulance_count = requested_ambulance_count()
    except ValueError as exc:
        return render_dashboard(get_active_scenario(), str(exc), 400)
    return render_dashboard(get_active_scenario(), ambulance_count=ambulance_count)


@app.get("/logo.png")
def logo_file():
    return send_from_directory(PROJECT_DIR, "logo.png")


@app.get("/search-trace")
def search_trace_page():
    return render_tab_page(
        "Search Algorithm Trace Studio",
        "Direct step-by-step comparison of A*, Risk-Aware A*, BFS, DFS, Greedy, and Hill Climbing.",
        "search",
    )


@app.get("/csp-solver")
def csp_solver_page():
    return render_tab_page(
        "Constraint Satisfaction (CSP) Solver",
        "Ambulance allocation with MRV heuristic, capacity constraints, and backtracking search.",
        "csp",
    )


@app.get("/ml-studio")
def ml_studio_page():
    return render_tab_page(
        "Machine Learning & Fuzzy Studio",
        "Supervised classifiers trained on FEMA disaster declarations and Fuzzy uncertainty inference.",
        "ml",
    )


@app.get("/analytics")
def analytics_page():
    return render_tab_page(
        "Mission Analytics & Trade-off Studio",
        "Quantitative algorithm comparisons for step count, hazard exposure, and rescue optimality.",
        "analytics",
    )


@app.post("/dashboard")
def custom_dashboard():
    try:
        scenario = scenario_from_form(request.form)
    except ValueError as exc:
        return render_dashboard(DEFAULT_SCENARIO, str(exc), 400)

    try:
        ambulance_count = requested_ambulance_count()
    except ValueError as exc:
        return render_dashboard(scenario, str(exc), 400)
    return render_dashboard(scenario, ambulance_count=ambulance_count)


@app.get("/simulation")
def simulation_page():
    result = run_simulation()
    return render_template_string(SIMULATION_HTML, result=result), 200 if result["ok"] else 500


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "AIDRA Flask API"})


@app.route("/api/scenario/generate", methods=["GET", "POST"])
def generate_scenario_api():
    try:
        size = int(request.args.get("size", 10))
        victims = int(request.args.get("victims", 4))
        hospitals = int(request.args.get("hospitals", 2))
        hazards = request.args.get("hazards", "medium")
        blocks = request.args.get("blocks", "medium")
        scen = generate_random_scenario(size, victims, hospitals, hazards, blocks)
        return jsonify({"ok": True, "scenario": scen})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/dashboard", methods=["GET", "POST"])
def dashboard_api():
    try:
        ambulance_count = requested_ambulance_count()
        event = request.args.get("event")
        preset = request.args.get("preset")

        if request.method == "GET" and request.args.get("reset"):
            clear_active_scenario()
            return jsonify(build_dashboard_data(DEFAULT_SCENARIO, ambulance_count=ambulance_count))

        if request.method == "GET":
            scenario = get_active_scenario()
        else:
            body = request.get_json(silent=True) or {}
            scenario = {k: v for k, v in body.items() if k not in ("ambulance_count", "event", "preset")}
            if not scenario:
                scenario = get_active_scenario()
            event = event or body.get("event")
            preset = preset or body.get("preset")

        payload = build_dashboard_data(scenario, event=event, preset=preset, ambulance_count=ambulance_count)
        set_active_scenario(payload["scenario"])
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/dashboard/form")
def dashboard_form_api():
    try:
        scenario = scenario_from_form(request.form)
        payload = build_dashboard_data(
            scenario,
            ambulance_count=requested_ambulance_count(),
        )
        set_active_scenario(scenario)
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/simulation")
def simulation_api():
    result = run_simulation()
    return jsonify(result), 200 if result["ok"] else 500


if __name__ == "__main__":
    app.run(debug=True)
