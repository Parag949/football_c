# 🏆 Hackathon Pitch, Presentation & Viva Guide
### Project Title: Computer Vision-Assisted Spatial Analytics and Geometric Algorithm Pipelines for Real-Time Football Match Reconstruction

---

## ⚡ 1. The 30-Second Elevator Pitch

> *"Current professional football analytics relies on multi-million dollar stadium camera arrays or wearable GPS vests that amateur clubs, academies, and TV viewers cannot access.*
> 
> *Our project solves this by taking **any single standard broadcast or panoramic video feed** and reconstructing an **interactive 2D and AR tactical radar in real time**. Using a lightweight YOLO detector coupled with multi-regime camera homography, we extract exact metric pitch coordinates and apply **classical computational geometry and spatial analytics**—specifically **Graham Scan** for defensive compactness, **Fortune's Voronoi Tessellation** for territorial dominance, **Continuous Gaussian Pitch Control** for space influence, and **Expected Threat ($xT$)** to value passing danger.*
> 
> *The entire system runs at **60+ FPS** in the browser, rendering live synchronized in-video augmented overlays and a 2D minimap side-by-side with zero specialized tracking hardware."*

---

## ⏱️ 2. The 2-Minute Live Demo Script (Step-by-Step)

When the judges say *"Show us the project"*, follow this script:

### Step 1: Open the Workstation (0:00 - 0:30)
- **Show the Dual Viewports**:
  - *"On the left, you see the **Broadcast Match Feed** with real-time AR overlays.*
  - *On the right, you see the **2D Tactical Pitch Radar**, calibrated to FIFA’s standard $105\text{m} \times 68\text{m}$ pitch dimensions.*
  - *Notice how every single player detected on the broadcast camera is mapped in real time to metric pitch turf."*

### Step 2: Hit "▶ Play Live" or Drag Scrubber (0:30 - 1:00)
- *"Watch the synchronized movement. As players shift on the pitch, our computational pipelines execute simultaneously:*
  1. **The Block Polygon**: *The **Defensive Convex Hull** computed using the Graham Scan algorithm. Its surface area tells a coach instantly whether their defensive block is compact or stretched.*
  2. **The Space Partition**: *A **Voronoi Partition** showing pitch dominance percentage. Team Blue currently controls $53\%$ of the territory.*
  3. **Continuous Pitch Control & xT**: *Radial Gaussian influence fields showing passing lanes and spatial threat transition value ($xT$)."*

### Step 3: Interactive 2D Tactics Board (1:00 - 1:30)
- Switch to Tab 3: **2D Tactics Board**.
- *"Coaches can import the exact live player positions from any video frame into an interactive tactics board. They can drag attackers and defenders, test 'what-if' runs, simulate offside lines with the Ghost Runner (FIFA Law 11 compliant), and watch team formations (e.g. 4-3-3 vs 4-4-2) adapt dynamically."*

### Step 4: Show the Empirical Benchmark Lab (1:30 - 2:00)
- Scroll down to the **Algorithm Complexity & Benchmark Suite**.
- *"We validated the asymptotic boundaries. For 22 players, all geometry algorithms execute in under **$3\text{ milliseconds}$** combined, easily surpassing the 33-millisecond threshold required for 30+ FPS broadcast television."*

---

## 🧠 3. Project Understanding in Plain English

### Is this a Machine Learning project or an Algorithm / DAA project?
**Answer**: It is a **Hybrid Engineering Pipeline where ML acts as the "Sensor" and DAA acts as the "Brain".**
- **Machine Learning (Perception)**: YOLOv8 detects bounding boxes, cleats, and ball candidates in camera pixels.
- **Computer Vision (Geometry)**: Multi-regime camera calibration (Planar, Universal Radial Distortion, Thin-Plate Spline) translates camera pixels into metric $(X, Y)$ coordinates in meters.
- **DAA & Computational Geometry (The Core Engine)**: Raw metric coordinates feed into:
  1. Convex Hull algorithms ($O(N \log N)$) to quantify defensive cohesion.
  2. Voronoi sweep-line partitions ($O(N \log N)$) clipped by Sutherland-Hodgman to quantify pitch territory.
  3. Continuous Gaussian Pitch Control and $16 \times 12$ Expected Threat ($xT$) matrix evaluation.
  4. FIFA Law 11 Ghost Runner offside line projection and tactical distance measurement.

---

## 📐 4. Deep-Dive on Core Algorithms

### Algorithm 1: Defensive Cohesion via Convex Hull
* **Tactical Meaning**: Defending teams must keep outfield players in a "compact block". If the defensive polygon expands beyond $1,200\,\text{m}^2$, spaces open between the lines for opponent playmakers.
* **Algorithms Implemented**:
  1. **Graham Scan** ($O(N \log N)$):
     - Finds lowest $y$-coordinate point $P_0$.
     - Sorts all remaining points by polar angle with $P_0$ in $O(N \log N)$.
     - Eliminates non-left turns using the 2D cross product:
       $$\text{Cross Product} = (x_2 - x_1)(y_3 - y_1) - (y_2 - y_1)(x_3 - x_1)$$
  2. **Jarvis March / Gift-Wrapping** ($O(N \cdot H)$):
     - Output-sensitive: runtime depends on $H$ (number of hull vertices).
* **Compactness Metric**:
  $$\text{Compactness} = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2} \in (0, 1]$$

### Algorithm 2: Spatial Dominance via Voronoi Tessellation
* **Tactical Meaning**: Pitch Control. If a ball is kicked to point $(x, y)$, which player reaches it first assuming equal running speed?
* **Algorithm**:
  - **Fortune's Sweep-Line Algorithm** ($O(N \log N)$): Maintains a parabolic beach line in a balanced BST.
  - **Sutherland-Hodgman Polygon Clipping**: Clips infinite Voronoi rays to the $105\text{m} \times 68\text{m}$ pitch rectangle.
* **Metric Produced**: Percentage of total pitch area ($7,140\,\text{m}^2$) dominated by Team A vs Team B.

### Algorithm 3: Continuous Pitch Control & Expected Threat ($xT$)
* **Continuous Pitch Control**: Computes radial influence fields:
  $$I_i(x, y) = \exp\left(-\frac{\|\mathbf{p}_i - (x, y)\|^2}{2 \sigma^2}\right)$$
* **Expected Threat ($xT$)**: Evaluates spatial transition matrices over a $16 \times 12$ pitch grid to quantify the incremental probability that a pass into that territory leads to a goal within the next sequence of actions.
