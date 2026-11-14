# 📊 Curated Football Data Sources & Platforms

A directory of football datasets, APIs, and raw data sources categorized by format and use case for computer vision, tactical analytics, and spatiotemporal modeling.

---

## 1. Match Stats, Results & Historical Archives (CSV, Excel)

| Platform | Formats | Primary Use Cases | Link |
| :--- | :--- | :--- | :--- |
| **FBref** | CSV, Excel, HTML | Advanced team/player statistics, passing networks, defensive actions, xG, and possession metrics (powered by Opta). | [fbref.com](https://fbref.com/) |
| **Football-Data.co.uk** | CSV, Excel (.xls/.xlsx) | Clean historical match results, betting odds, match dates, referee statistics, and multi-decade European league data. | [football-data.co.uk](https://www.football-data.co.uk/) |
| **DataHub Football Collection** | CSV, JSON | Open-source, ready-to-use tabular dumps covering major European leagues, historical World Cups, stadium information, and club profiles. | [datahub.io/collections/football](https://datahub.io/collections/football) |

---

## 2. Player Profiling & Advanced Spatial Metrics (JSON, Web Scraping)

| Platform | Formats | Primary Use Cases | Link |
| :--- | :--- | :--- | :--- |
| **Transfermarkt** | Structured HTML / JSON | Player market values, historical transfers, contract expirations, player agents, and injury histories. | [transfermarkt.com](https://www.transfermarkt.com/) |
| **Understat** | JSON (API endpoints) | Detailed shot-level data (shot maps), Expected Goals (xG), Expected Assists (xA), and spatial coordinates of attempts across Europe's top 5 leagues. | [understat.com](https://understat.com/) |
| **WhoScored / Sofascore** | Dynamic JSON / Web Data | In-depth post-match player performance ratings, touch heatmaps, and tactical match dashboards. | [whoscored.com](https://www.whoscored.com/) / [sofascore.com](https://www.sofascore.com/) |

---

## 3. Video, Audio & Computer Vision Datasets (MP4, High-Res Images)

| Dataset | Formats | Primary Use Cases | Link |
| :--- | :--- | :--- | :--- |
| **SoccerNet** | MP4, Audio, JSON annotations | Computer vision, action spotting, camera calibration, player re-identification, deep learning tracking, and crowd audio analysis. | [soccer-net.org](https://www.soccer-net.org/) |
| **Kaggle Football Vision** | MP4, JPEG/PNG, CSV | Labeled datasets for object detection (detecting ball and players using YOLO/OpenCV), trajectory prediction, and event classification. | [kaggle.com](https://www.kaggle.com/) |

---

## 4. Spatiotemporal & Event-Level Open Datasets (JSON, Parquet, Python)

| Platform | Formats | Primary Use Cases | Link |
| :--- | :--- | :--- | :--- |
| **StatsBomb Open Data** | JSON, CSV (`statsbombpy`) | High-detail event logs, 360-degree freeze-frame spatial data (coordinates of all players in frame at pass/shot instances), and historical international tournaments. | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| **Metrica Sports Open Tracking** | CSV, JSON (25Hz raw XY) | Full-pitch 2D tracking data, Voronoi/Dominant region modeling, speed/acceleration calculation, and defensive pitch control analysis. | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| **Wyscout (PlayeRank)** | JSON, Figshare | Millions of multi-league spatiotemporal match events covering passes, duel outcomes, and defensive maneuvers. | [figshare.com](https://figshare.com/) |

---

## 5. Developer APIs & Live Feeds (REST, JSON, XML)

| Provider | Protocol / Format | Primary Capabilities | Link |
| :--- | :--- | :--- | :--- |
| **API-Football** | REST API (JSON) | Live score updates, league standings, player statistics, fixture lists, team squad sheets, and head-to-head comparisons. | [api-football.com](https://api-football.com/) |
| **Sportmonks** | REST API (JSON) | Fast in-game event feeds, betting odds APIs, line-ups, and statistical historical records. | [sportmonks.com](https://www.sportmonks.com/) |
