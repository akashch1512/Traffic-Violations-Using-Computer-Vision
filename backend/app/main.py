from fastapi import FastAPI
from app.core.config import settings

from app.api import vehicle, violations, cameras, tracking, analytics, license_plate

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Include the routers (wiring them to the app)
app.include_router(vehicle.router, prefix="/vehicles", tags=["Vehicles"])
app.include_router(cameras.router, prefix="/cameras", tags=["Cameras"])
app.include_router(violations.router, prefix="/violations", tags=["Violations"])
app.include_router(tracking.router, prefix="/tracking", tags=["Tracking"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(license_plate.router, prefix="/license-plate", tags=["License Plate"])

@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": f"Welcome to the {settings.PROJECT_NAME}",
        "architecture_status": "All modules connected successfully."
    }