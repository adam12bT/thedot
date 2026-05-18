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

## 📂 How to use

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

## 📊 Charts included

| # | Chart |
|---|-------|
| 1 | Events by Status |
| 2 | Events by Day of Week |
| 3 | Events per Month (line + fill) |
| 4 | Events by Activity Type |
| 5 | Events by Start Hour |
| 6 | Top 10 Most-Used Rooms |
| 7 | Top 15 Organizations |
| 8 | Avg Participants by Activity Type |
| 9 | Cancellation Rate by Room |
| 10 | Rooms by Total Reservation Hours |
| 11 | Participant Distribution (box + hist) |
| 12 | Heatmap: Weekday × Hour |

---

## 🔍 Column auto-detection

The app automatically maps these column name variants:

| Concept | Accepted names |
|---------|---------------|
| Title | `title`, `event_name`, `name`, `nom` |
| Start | `startTime`, `start_time`, `start` |
| End | `endTime`, `end_time`, `end` |
| Status | `status`, `statut` |
| Visibility | `visibility`, `type` |
| Room | `event_proposals.space.name`, `room`, `salle` |
| Email | `organizer.email`, `email` |
| First name | `organizer.firstName`, `firstName` |
| Last name | `organizer.lastname`, `lastName` |
| Org | `organizer.organization.name`, `organization` |
| Participants | `participantNb`, `participants` |
| Policy | `policy`, `declaration`, `signed` |
| Theme | `theme`, `thème`, `category` |

---

## 💾 Exports

- **cleaned_events.xlsx** — styled Excel with dropdowns, frozen header, table format, SUMMARY sheet
- **removed_rows.xlsx** — sheets for empty rows, duplicates, negative durations
- **filtered_events.csv** — the currently filtered data as CSV
