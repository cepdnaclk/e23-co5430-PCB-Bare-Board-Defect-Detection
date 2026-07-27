import os
import sys
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="MicroInspect UI")

# Base paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
# demo.py saves outputs to the parent of PROJECT_ROOT (CVProject/outputs)
OUTPUTS_DIR = PROJECT_ROOT.parent / "outputs"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "static").mkdir(parents=True, exist_ok=True)

# Mount outputs directory to serve result images
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount static directory for HTML/CSS/JS
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/")
def read_root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

@app.post("/analyze")
async def analyze(
    method: str = Form(...),
    test_img: UploadFile = File(...),
    template_img: UploadFile = File(None)
):
    if method not in ["dl", "classical", "classical_topological"]:
        raise HTTPException(status_code=400, detail="Invalid method. Choose 'dl', 'classical', or 'classical_topological'.")
        
    if method in ["classical", "classical_topological"] and not template_img:
        raise HTTPException(status_code=400, detail="Template image is required for classical methods.")
        
    # Save uploaded test image
    test_img_path = TEMP_DIR / test_img.filename
    with open(test_img_path, "wb") as buffer:
        shutil.copyfileobj(test_img.file, buffer)
        
    template_img_path = None
    if method in ["classical", "classical_topological"] and template_img:
        template_img_path = TEMP_DIR / template_img.filename
        with open(template_img_path, "wb") as buffer:
            shutil.copyfileobj(template_img.file, buffer)
            
    # Build command
    demo_script = PROJECT_ROOT / "scripts" / "demo.py"
    cmd = [sys.executable, str(demo_script), "--test_img", str(test_img_path), "--method", method]
    
    if template_img_path:
        cmd.extend(["--template_img", str(template_img_path)])
        
    # Run subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise HTTPException(status_code=500, detail=f"Error running inference: {e.stderr}")
        
    # Determine outputs
    test_img_name = test_img_path.stem
    outputs = {}
    
    if method == "dl":
        result_path = f"/outputs/dl/{test_img_name}/{test_img_name}_result.jpg"
        outputs["result"] = result_path
    elif method == "classical":
        outputs["result"] = f"/outputs/classical/{test_img_name}/{test_img_name}_result.jpg"
        outputs["mask"] = f"/outputs/classical/{test_img_name}/{test_img_name}_mask.jpg"
        outputs["aligned"] = f"/outputs/classical/{test_img_name}/{test_img_name}_aligned.jpg"
    elif method == "classical_topological":
        outputs["result"] = f"/outputs/classical_topological/{test_img_name}/{test_img_name}_result.jpg"
        outputs["mask"] = f"/outputs/classical_topological/{test_img_name}/{test_img_name}_mask.jpg"
        outputs["aligned"] = f"/outputs/classical_topological/{test_img_name}/{test_img_name}_aligned.jpg"
        
    return {
        "status": "success",
        "method": method,
        "outputs": outputs,
        "logs": result.stdout
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
