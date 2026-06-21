# 🛣️ AI Road Damage Assessment System

A full-stack web application that uses **OpenCV**, **Scikit-learn**, and **Google Gemini AI** to analyse road photographs and generate professional inspection reports.

---

## 🚀 Quick Start

### 1. Clone / unzip and enter the project folder
```bash
cd road_damage_app
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Generate training data and train the ML model
```bash
python generate_dataset.py   # Creates road_damage_dataset.csv
python ml_model.py           # Trains and saves the Random Forest model
```

### 6. Run the Flask application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 📁 Project Structure

```
road_damage_app/
├── app.py                  # Flask routes & application entry point
├── analyzer.py             # OpenCV image processing & feature extraction
├── ml_model.py             # Scikit-learn Random Forest classifier
├── report_generator.py     # LangChain + Gemini AI report generation
├── generate_dataset.py     # Synthetic training data generator
├── requirements.txt
├── .env.example
├── models/                 # Saved model artifacts (auto-created)
├── uploads/                # Temporary image storage (auto-created)
├── templates/
│   └── index.html          # Main dashboard UI
└── static/
    ├── style.css       # Design system & component styles
    └── js/app.js           # Frontend logic (drag-drop, API, rendering)
```

---

## 🔬 Analysis Pipeline

| Step | Module | Technology |
|------|--------|------------|
| Image pre-processing | `analyzer.py` | OpenCV — Gaussian blur, adaptive threshold |
| Feature extraction | `analyzer.py` | Canny edges, Laplacian, contour detection |
| Priority classification | `ml_model.py` | Scikit-learn Random Forest (200 trees) |
| Cost estimation | `report_generator.py` | Rule-based formula with priority rates |
| Report generation | `report_generator.py` | LangChain + Google Gemini 1.5 Flash |

---

## 📊 Extracted Features

| Feature | Method |
|---------|--------|
| `damage_percentage` | Adaptive thresholding → pixel ratio |
| `num_damaged_regions` | Contour detection with area filter |
| `crack_density` | Canny edge pixel ratio |
| `texture_roughness` | Laplacian variance (normalised) |
| `dark_surface_percentage` | Grayscale intensity threshold |

---

## 🎯 Priority Levels

| Level | Typical Repair Window | Cost Rate |
|-------|----------------------|-----------|
| Low Priority | 12–24 months | $15/m² |
| Medium Priority | 6–12 months | $40/m² |
| High Priority | 1–3 months | $90/m² |
| Critical Priority | Within 1–2 weeks | $175/m² |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask 3.0
- **Computer Vision**: OpenCV 4.8
- **Machine Learning**: Scikit-learn 1.3 (Random Forest)
- **AI Reports**: LangChain 0.1 + Google Gemini 1.5 Flash
- **Frontend**: Bootstrap 5.3, Vanilla JS, marked.js (Markdown)

---

## 📝 Notes

- The system works **without a Gemini API key** — it falls back to a structured template report.
- Uploaded images are processed in memory and deleted after analysis.
- The ML model auto-trains on first run if model files are missing.
