import unittest

from app import app, _plan_rescue_dispatches, create_environment, parse_ambulance_count, serialize_grid
from agent import choose_best_victim
from search import a_star, a_star_risk, bfs


class SearchPlanningTests(unittest.TestCase):
    def test_a_star_matches_bfs_shortest_distance_around_obstacles(self):
        env = create_environment({
            "size": 6,
            "base": [0, 0],
            "hospitals": [[5, 5]],
            "victims": [{"id": 1, "x": 4, "y": 4, "severity": "critical"}],
            "blocked_cells": [[1, 1], [1, 2], [2, 1]],
        })

        self.assertEqual(len(a_star(env, (0, 0), (4, 4))), len(bfs(env, (0, 0), (4, 4))))

    def test_risk_aware_a_star_avoids_costly_hazard_when_detour_is_cheaper(self):
        env = create_environment({
            "size": 5,
            "base": [0, 0],
            "hospitals": [[4, 4]],
            "victims": [{"id": 1, "x": 3, "y": 4, "severity": "critical"}],
            "risk_zones": [{"x": 2, "y": 2, "level": 3}],
        })

        shortest = a_star(env, (2, 0), (2, 4))
        safer = a_star_risk(env, (2, 0), (2, 4), risk_weight=3)

        self.assertEqual(len(shortest), 5)
        self.assertNotIn((2, 2), safer)
        self.assertGreater(len(safer), len(shortest))

    def test_victim_selector_handles_fully_blocked_start(self):
        env = create_environment({
            "size": 4,
            "base": [0, 0],
            "hospitals": [[3, 3]],
            "victims": [{"id": 1, "x": 2, "y": 2, "severity": "critical"}],
            "blocked_cells": [[1, 0], [0, 1]],
        })

        self.assertEqual(choose_best_victim(env, env.base), (None, None))


class RescueDispatchTests(unittest.TestCase):
    def setUp(self):
        self.scenario = {
            "size": 8,
            "base": [0, 0],
            "hospitals": [[7, 7], [0, 7]],
            "victims": [
                {"id": 1, "x": 2, "y": 2, "severity": "critical"},
                {"id": 2, "x": 3, "y": 2, "severity": "moderate"},
                {"id": 3, "x": 5, "y": 3, "severity": "minor"},
                {"id": 4, "x": 5, "y": 5, "severity": "critical"},
            ],
        }

    def test_map_uses_stable_endpoint_labels_and_victim_severity(self):
        grid = serialize_grid(create_environment(self.scenario))

        self.assertEqual(grid[0][0]["marker_label"], "BASE")
        self.assertEqual(grid[7][7]["marker_label"], "H1")
        self.assertEqual(grid[0][7]["marker_label"], "H2")
        self.assertEqual(grid[2][2]["marker_label"], "V1")
        self.assertEqual(grid[2][2]["victim_severity"], "critical")
        self.assertEqual(grid[3][2]["victim_severity"], "moderate")
        self.assertEqual(grid[5][3]["victim_severity"], "minor")

    def test_overview_lists_map_endpoints_and_all_application_pages(self):
        response = app.test_client().get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for label in (
            "Map endpoints",
            "Base BASE",
            "Hospital H1",
            "Victim V1",
            'href="/ml-studio"',
            'href="/simulation"',
            "victim-critical",
            "victim-moderate",
            "victim-minor",
        ):
            self.assertIn(label, html)

    def test_one_ambulance_repeats_trips_from_its_last_hospital(self):
        dispatches, ambulances = _plan_rescue_dispatches(create_environment(self.scenario), 1)

        self.assertEqual(len(dispatches), len(self.scenario["victims"]))
        self.assertEqual({item["ambulance"] for item in dispatches}, {"Ambulance 1"})
        for previous, following in zip(dispatches, dispatches[1:]):
            self.assertEqual(following["from"], previous["hospital"])
        self.assertEqual(ambulances[0]["rescued"], len(self.scenario["victims"]))

    def test_more_ambulances_share_the_rescue_queue(self):
        dispatches, ambulances = _plan_rescue_dispatches(create_environment(self.scenario), 3)

        self.assertEqual(len(dispatches), len(self.scenario["victims"]))
        self.assertEqual(len({item["ambulance"] for item in dispatches}), 3)
        self.assertEqual(sum(item["rescued"] for item in ambulances), len(self.scenario["victims"]))

    def test_fleet_size_is_limited_to_one_through_six(self):
        self.assertEqual(parse_ambulance_count("1"), 1)
        with self.assertRaises(ValueError):
            parse_ambulance_count("0")
        with self.assertRaises(ValueError):
            parse_ambulance_count("7")

    def test_dashboard_api_uses_selected_fleet_and_returns_all_trips(self):
        response = app.test_client().get("/api/dashboard?ambulance_count=1")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["ambulance_count"], 1)
        self.assertEqual(data["decision"]["saved"], len(data["victims"]))
        self.assertEqual(len(data["dispatches"]), len(data["victims"]))
        victim_cells = [
            cell
            for row in data["grid"]
            for cell in row
            if "victim" in cell["terrain_type"]
        ]
        self.assertEqual(len(victim_cells), len(data["victims"]))
        self.assertTrue(all(item["ambulance"] == "Ambulance 1" for item in data["dispatches"]))
        for previous, following in zip(data["dispatches"], data["dispatches"][1:]):
            self.assertEqual(following["from"], previous["hospital"])

    def test_dashboard_api_rejects_out_of_range_fleet_size(self):
        response = app.test_client().get("/api/dashboard?ambulance_count=7")

        self.assertEqual(response.status_code, 400)
        self.assertIn("between 1 and 6", response.get_json()["error"])

    def test_dashboard_reports_weighted_route_cost_against_alternative_algorithms(self):
        response = app.test_client().get("/api/dashboard")
        data = response.get_json()
        algorithms = {item["name"]: item for item in data["algorithms"]}
        risk_aware = algorithms["Risk-Aware A*"]
        reachable_alternatives = [
            item for name, item in algorithms.items()
            if name != "Risk-Aware A*" and item["reached"]
        ]

        self.assertEqual(data["route_risk_weight"], 3)
        for item in algorithms.values():
            if item["reached"]:
                self.assertEqual(
                    item["weighted_cost"],
                    item["length"] + data["route_risk_weight"] * item["risk"],
                )
        self.assertTrue(reachable_alternatives)
        self.assertLessEqual(
            risk_aware["weighted_cost"],
            min(item["weighted_cost"] for item in reachable_alternatives),
        )


if __name__ == "__main__":
    unittest.main()
