# License Plate Detection

Parking management system that detects license plates from uploaded images using YOLOv5 and TrOCR, records vehicle entry/exit in MySQL, and calculates parking fares.

## Tech stack

- **Backend:** Flask, Gunicorn
- **ML:** YOLOv5 (plate detection), TrOCR (character recognition), PyTorch (CPU)
- **Database:** MySQL 8
- **Frontend:** HTML templates + static CSS

## Project structure

```
backend/
  app.py              # Flask routes and parking logic
  model/
    LPD2.py           # Plate detection and OCR pipeline
    yolov5/           # Local YOLOv5 hub (vendored)
  requirements.txt
frontend/
  templates/          # login, upload, logs pages
  static/
docker/               # entrypoint, MySQL init SQL
models/               # Place plate_detection.pt here (not in Git)
```

## Features

- Admin login with hashed passwords
- Vehicle **entry** and **exit** via image upload
- Automatic plate recognition (when model weights are present)
- Parking logs, dashboard stats, and plate search
- Fare rules: free under 30 minutes, 1000 MMK after
- Health check endpoint at `/health`

## Prerequisites

The trained YOLO weights file **`plate_detection.pt`** is not stored in GitHub. Copy your trained model to:

```
models/plate_detection.pt
```

On first inference, TrOCR models are downloaded from Hugging Face (~1–2 GB).

## Quick start (Docker)

```bash
cp .env.example .env   # change MYSQL_ROOT_PASSWORD
# copy models/plate_detection.pt before running inference
docker compose up -d --build
```

Open **http://localhost:5000**

Default login: `admin` / `admin123`

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MYSQL_HOST` | MySQL hostname | `db` (Docker) |
| `MYSQL_USER` | Database user | `root` |
| `MYSQL_PASSWORD` | Database password | `changeme` |
| `MYSQL_DATABASE` | Database name | `parking_db` |
| `PLATE_MODEL_PATH` | Path to YOLO weights | `/app/models/plate_detection.pt` |
| `LPD_PORT` | Host port | `5000` |
| `FLASK_DEBUG` | Debug mode | `false` |
| `GUNICORN_WORKERS` | Worker count | `1` |

## API routes

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/health` | Health and DB status |
| POST | `/login` | Authenticate user |
| POST | `/upload-entry` | Record vehicle entry |
| POST | `/upload-exit` | Record vehicle exit and fare |
| GET | `/get-stats` | Dashboard statistics |
| GET | `/get-logs` | Recent parking logs |
| GET | `/search-car?plate=` | Search by plate |

## Local development (without Docker)

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Start MySQL locally and set MYSQL_* env vars
python app.py
```

## Deploy on a Linux server

1. Clone repo, add `models/plate_detection.pt`, copy `.env.example` to `.env`
2. Run `docker compose up -d --build`
3. First build is large (PyTorch + dependencies); allow 15–30+ minutes
4. Expose port 5000 or put nginx in front for HTTPS

Volumes persist MySQL data, uploads, and ML model cache.
