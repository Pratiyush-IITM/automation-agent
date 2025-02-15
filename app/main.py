from fastapi import FastAPI, HTTPException, Query
import subprocess
import os
from app.tasks import execute_task
from app.utils import read_file_content

app = FastAPI()

DATA_DIR = "/data"  # Restrict file access to this directory

@app.post("/run")
async def run_task(task: str = Query(..., description="Plain-English task description")):
    """Executes a given task using LLM and internal scripts."""
    try:
        result = execute_task(task)
        return {"status": "success", "output": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/read")
async def read_file(path: str):
    """Reads and returns the content of a file inside /data directory."""
    file_path = os.path.join(DATA_DIR, os.path.basename(path))
    content = read_file_content(file_path)
    
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {"file": path, "content": content}
