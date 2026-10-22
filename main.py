from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from diarization import process_audio
import shutil
import uuid
import os
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up temp directory and mount as static
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory="temp"), name="static")


def cleanup_files(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


# Plain `def` (not async): process_audio is blocking CPU work, so FastAPI runs it in
# a worker thread instead of freezing the event loop (and /static) for every request.
@app.post("/diarize")
def diarize(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        # Save uploaded audio to temp file
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)

        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process with PyAnnote
        result = process_audio(audio_path)

        # Schedule cleanup of all generated files
        background_tasks.add_task(cleanup_files, audio_path)

        # Build accessible URLs
        image_url = f"{os.path.basename(result['image_path'])}"
        rttm_url = f"{os.path.basename(result['rttm'])}"

        return JSONResponse({
            "image_url": image_url,
            "rttm_url": rttm_url
        })

    except Exception as e:
        logger.exception("Error in diarization")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
