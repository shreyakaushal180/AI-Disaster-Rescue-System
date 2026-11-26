# AIDRA Project Notes

## 1. Project Overview

AIDRA (Adaptive Intelligent Disaster Response Agent) is an academic demonstration of an autonomous disaster-rescue decision-support system. It uses a grid-based disaster environment to identify victims, select priorities, allocate ambulances, calculate safe routes, and show rescue operations in a Flask web command center.

The project combines five AI concepts:

1. Search and pathfinding algorithms
2. Constraint Satisfaction Problems (CSP)
3. Multi-factor agent-based victim selection
4. Machine learning ensemble classification
5. Fuzzy logic for uncertain conditions

The system is not limited to one algorithm. It provides a comparison and explanation view so users can observe how different techniques affect rescue decisions.

---

## 2. Core Problem

During a disaster, rescue teams must decide:

- Which victim should be rescued first?
- Which route is fastest and safest?
- Which ambulance should handle the victim?
- How should plans change when roads are blocked or hazards increase?
- How should victims be sent to reachable hospitals?

AIDRA represents the disaster area as a grid containing:

- A rescue base
- Hospitals
- Victims
- Blocked cells
- Low, medium, or high-risk hazard zones
- Rescue ambulances

---

## 3. System Features

### Live Command Center

- Configurable 1-6 ambulance fleet
- Multiple presets and custom disaster scenarios
- Animated rescue dispatches
- Victim, hospital, base, hazard, and blocked-cell display
- Rescue notifications and activity history
- Dynamic events such as road blockage, fire/risk increase, new victims, and ambulance loss
- Random scenario generation

### Algorithm Comparison

The project compares:

- Standard A*
- Risk-Aware A*
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Greedy Best-First Search
- Hill Climbing

The weighted route cost is:

`Steps + (3 x cumulative path risk)`

Risk-Aware A* adds the hazard cost to route planning, while standard A* prioritizes the shortest route.

### Machine Learning

The project trains three classifiers using the FEMA disaster dataset:

- k-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree

The models predict:

- LOW
- MEDIUM
- HIGH

priority levels. The dashboard can display model accuracy, precision, recall, F1-score, and confusion matrix information.

### Fuzzy Logic

Fuzzy logic categorizes:

- Distance: Near, Medium, or Far
- Risk: Low, Medium, or High
- Blockage: Unlikely, Possible, or Likely

It produces a support decision describing whether rescue is urgent or risky under uncertain conditions.

### CSP Resource Allocation

The CSP module allocates victims to ambulances with:

- Maximum two victims per ambulance
- Severity-based ordering
- MRV-style variable selection
- Least-constrained ambulance ordering
- Backtracking
- Capacity and backtracking trace

---

## 4. Important Source Files

| File | Responsibility |
|---|---|
| `app.py` | Flask application, dashboard, API routes, scenarios, dispatch planning, and page rendering |
| `environment.py` | Grid, cells, victims, hospitals, base, blocked cells, and hazard zones |
| `search.py` | BFS, DFS, A*, Risk-Aware A*, Greedy Best-First, and Hill Climbing |
| `agent.py` | Victim priority scoring and hospital selection |
| `csp.py` | Ambulance allocation using CSP, MRV-style selection, and backtracking |
| `ml_model.py` | FEMA dataset processing and classifier training/prediction |
| `fuzzy.py` | Fuzzy distance, risk, and blockage inference |
| `main.py` | Standalone terminal simulation |
| `test_rescue_planning.py` | Search, dispatch, API, route, and fleet-size tests |
| `DisasterDeclarationsSummaries.csv` | FEMA historical disaster dataset |
| `requirements.txt` | Python dependencies |

---

## 5. Application Flow

1. A disaster scenario is loaded or created.
2. The environment builds the grid and places victims, hospitals, obstacles, and hazards.
3. The agent evaluates each reachable victim using severity, distance, and path risk.
4. The highest-priority victim is selected.
5. Available ambulances receive the rescue queue.
6. Risk-Aware A* plans a route to the victim.
7. The ambulance takes the victim to a reachable hospital.
8. The next trip begins from the hospital.
9. The system updates when scenario events occur.
10. Results are displayed in the dashboard, analysis pages, and simulation trace.

---

## 6. Team Division for Five People

### Person 1 - Project Lead and Flask Integrator

**Main files:** `app.py`, `main.py`

Responsibilities:

- Coordinate the team and maintain the project timeline
- Integrate completed modules into the Flask application
- Maintain dashboard data flow and API routes
- Review pull requests and merge changes
- Prepare the final demo and presentation

**Deliverables:**

- Stable main application
- Scenario and dispatch integration
- Demo script
- Final project summary

### Person 2 - Environment and Search Algorithm Developer

**Main files:** `environment.py`, `search.py`

Responsibilities:

- Maintain the grid and cell model
- Implement and verify search algorithms
- Ensure blocked cells and risk zones are handled correctly
- Verify path reconstruction and algorithm edge cases
- Explain route cost and algorithm trade-offs

**Deliverables:**

- Tested search implementation
- Algorithm comparison data
- Environmental edge-case handling

### Person 3 - Rescue Agent and CSP Developer

**Main files:** `agent.py`, `csp.py`

Responsibilities:

- Implement victim priority scoring
- Select victims and hospitals
- Maintain ambulance capacity and assignment logic
- Implement and document backtracking and MRV-style behavior
- Optimize dispatch planning for 1-6 ambulances

**Deliverables:**

- Victim selection module
- Ambulance allocation trace
- Dispatch planning tests

### Person 4 - Machine Learning and Fuzzy Logic Developer

**Main files:** `ml_model.py`, `fuzzy.py`, `DisasterDeclarationsSummaries.csv`

Responsibilities:

- Process FEMA data and maintain features
- Train k-NN, Naive Bayes, and Decision Tree models
- Produce and validate model metrics
- Implement fuzzy categories and inference rules
- Document how ML and fuzzy decisions support the rescue agent

**Deliverables:**

- Working prediction interface
- Model metrics
- Fuzzy decision explanation
- Dataset and feature documentation

### Person 5 - Frontend, Testing, and Documentation

**Main files:** `test_rescue_planning.py`, `README.md`, `PROJECT_NOTES.md`

Responsibilities:

- Maintain the dashboard UI and display components
- Improve usability and page navigation
- Write and run unit/integration tests
- Document setup, features, limitations, and demo steps
- Capture screenshots or demonstrate the application

**Deliverables:**

- Responsive and understandable interface
- Passing test suite
- Setup and usage documentation
- Presentation evidence

---

## 7. Suggested Work Schedule

### Week 1

- Review project files and dependencies
- Assign Git branches and file ownership
- Establish shared naming and code-format conventions
- Run the existing project and tests

### Week 2

- Person 2: complete and test search/environment modules
- Person 3: complete victim scoring and CSP behavior
- Person 4: complete ML and fuzzy modules
- Person 1: finish API integration
- Person 5: prepare test cases and UI notes

### Week 3

- Integrate all modules
- Test dynamic events and different scenarios
- Fix bugs found during integration
- Complete the documentation

### Week 4

- Run complete end-to-end demonstrations
- Prepare screenshots or a video
- Practice the presentation
- Submit or present the project

---

## 8. Git and Collaboration Rules

- Use one branch per feature.
- Avoid multiple people editing the same function simultaneously.
- Person 1 reviews and merges completed work.
- Add tests with every functional change.
- Do not commit virtual environments, caches, or generated files.
- Use clear commit messages, for example:
  - `Add Risk-Aware A* route comparison`
  - `Fix ambulance fleet limit validation`
  - `Document FEMA ML workflow`

---

## 9. Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the web application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Run the terminal simulation:

```bash
python main.py
```

Run tests:

```bash
python -m unittest test_rescue_planning.py
```

Useful pages:

- `/` - Live command center
- `/search-trace` - Search algorithm comparison
- `/csp-solver` - Ambulance allocation
- `/ml-studio` - ML and fuzzy analysis
- `/analytics` - Route and risk analytics
- `/simulation` - Terminal simulation output

---

## 10. Demo Checklist

- Start the Flask server successfully
- Open the command center
- Explain the map endpoints and severity colors
- Start a 1-ambulance rescue
- Start a 3- or 6-ambulance rescue
- Compare Risk-Aware A* with Standard A*
- Trigger a blocked-road event
- Trigger a risk spike
- Generate a random scenario
- Open the ML Studio
- Show the FEMA model metrics
- Run the CSP solver trace
- Run the test suite

---

## 11. Presentation Structure

1. Problem statement
2. Project objective
3. AI techniques used
4. Environment and scenario
5. Priority and dispatch logic
6. Live web demonstration
7. Algorithm comparison
8. ML and fuzzy logic explanation
9. Testing and limitations
10. Future improvements

---

## 12. Possible Future Improvements

- Real-world maps and GPS coordinates
- Live disaster and weather API integration
- Machine-learning route prediction
- Real-time ambulance location tracking
- Time-dependent traffic and road conditions
- Multiple casualty types and medical resources
- Database-backed rescue history
- Authentication and role-based operator access
- Mobile emergency-responder interface
- Docker deployment
