---
title: Event Analytics
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.45.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 📊 Event Analytics Pipeline — Streamlit App

A fully dynamic, interactive web dashboard that processes event data from
`data.csv` (or `data.xlsx`) directly in your browser.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📂 How to Use

1. Run `streamlit run app.py`
2. In the **sidebar**, upload your `data.csv` or `data.xlsx`
3. Use the **filters** (Status, Activity Type, Year) to slice the data
4. Browse the **5 tabs**:
   - **📊 Charts** — 12 interactive visualisations
   - **🗂️ Data Table** — scrollable cleaned data
   - **🧹 Cleaning Report** — removed rows, duplicates, fixes
   - **📋 Statistics** — descriptive stats & pivot tables
   - **💾 Export** — download cleaned Excel, removed rows report, filtered CSV

---

## 📊 Charts Included

| # | Chart |
|---|-------|
| 1 | Events by Status |
| 2 | Events by Day of Week |
| 3 | Events per Month (line + fill) |
| 4 | Year-over-Year Trend |
| 5 | Events by Activity Type |
| 6 | Events by Start Hour |
| 7 | Top 10 Most-Used Rooms |
| 8 | Top 15 Organizations |
| 9 | Avg Participants by Activity Type |
| 10 | Cancellation Rate by Room |
| 11 | Rooms by Total Reservation Hours |
| 12 | Participant Distribution (box + histogram) |
| 13 | Heatmap: Weekday × Hour |

---

## 🔍 Column Auto-Detection

The app automatically maps these column name variants:

| Concept | Accepted Names |
|---------|----------------|
| Title | `title`, `event_name`, `name`, `nom` |
| Start | `startTime`, `start_time`, `start`, `début` |
| End | `endTime`, `end_time`, `end`, `fin` |
| Status | `status`, `statut`, `état` |
| Visibility | `visibility`, `type`, `visibilité` |
| Room | `event_proposals.space.name`, `room`, `salle`, `space` |
| Email | `organizer.email`, `email`, `courriel` |
| First name | `organizer.firstName`, `firstName`, `prenom`, `first_name` |
| Last name | `organizer.lastname`, `lastName`, `last_name` |
| Organisation | `organizer.organization.name`, `organization`, `organisation`, `org` |
| Participants | `participantNb`, `participants`, `participant_count` |
| Policy | `policy`, `declaration`, `signed` |
| Theme | `theme`, `thème`, `category`, `catégorie` |
| Booking date | `booking_date`, `bookingDate`, `reservation_date` |

---

## 🧹 Data Cleaning

The pipeline automatically handles:

- **Empty rows** — rows where all key fields are blank are removed and reported
- **Duplicates** — rows with identical event name, start/end time, and room are deduplicated
- **Negative durations** — start and end times are swapped and flagged
- **Timestamp fixes** — malformed hour/minute values are corrected
- **Participant outliers** — events with more than 1,000 participants are flagged and excluded from aggregate stats (but kept in the dataset)
- **Status normalisation** — `finished` → `Tenu`, `canceled`/`rejected` → `Annulé`, `Reporté` → `Annulé`
- **Space name normalisation** — room names are mapped to canonical French labels

---

## 💾 Exports

| File | Description |
|------|-------------|
| `cleaned_events.xlsx` | Styled Excel with dropdown validation, frozen header, table format, and a SUMMARY sheet |
| `removed_rows.xlsx` | Three sheets: empty rows, duplicates, negative-duration rows |
| `filtered_events.csv` | The currently filtered view as plain CSV |

---

## 🗂️ Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── tests/
    └── test_pipeline.py # Unit & integration tests
```

---

## ✅ Running Tests

```bash
pytest tests/ --cov=app --cov-report=term-missing
```

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
