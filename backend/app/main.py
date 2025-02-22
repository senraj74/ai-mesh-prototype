from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.segmentation import segment_mesh
from app.meshing import mesh_optimization
from pydantic import BaseModel

import openai
import trimesh
import os
import open3d as o3d
import numpy as np
import os
import shutil


#  Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

#  Initialize FastAPI
app = FastAPI()


#  Enable CORS for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

#  Define Request Model
class ChatRequest(BaseModel):
    user_input: str  # Expecting JSON with { "user_input": "..." }

@app.post("/chat/")
async def chatbot_interaction(request: ChatRequest):
    """
    Handles chatbot conversation using GPT-4 for AI meshing queries.
    """

    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": request.user_input}
    ]

    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5000
    )

    return {"response": response.choices[0].message.content.strip()}


@app.post("/upload-mesh/")
async def upload_mesh(file: UploadFile = File(...)):
    upload_dir = "backend/data/uploads/"
    output_dir = "backend/data/processed/"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    optimized_path = os.path.join(output_dir, f"optimized_{file.filename}.glb")

    #  Save File
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        #  Process Mesh
        segmented_mesh = segment_mesh(file_path)

        #  Save Optimized Mesh
        o3d.io.write_triangle_mesh(optimized_path, segmented_mesh)

        #  Debug: Check if the file was written
        if not os.path.exists(optimized_path):
            raise RuntimeError("Processed mesh file was not created!")

        file_size = os.path.getsize(optimized_path)
        if file_size < 1000:  # Check for small files
            raise RuntimeError(f"Processed mesh file is too small ({file_size} bytes)!")

        return {"message": " File processed!", "mesh_url": f"http://localhost:8000/download-mesh/{file.filename}.glb"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Serve Optimized Mesh
@app.get("/download-mesh/{filename}")
async def download_mesh(filename: str):
    file_path = f"backend/data/processed/optimized_{filename}"
    if not os.path.exists(file_path):
        return {"error": "Mesh not found!"}
    
    return FileResponse(file_path, filename=f"optimized_{filename}")

