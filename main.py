from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, Form, HTTPException , File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from visualization import chain
from codefix import code_fix_chain
from workflowmain import generate_case_study_v2
import tempfile
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/generate-visualization/")
async def generate_visualization(
    features: str = Form(...),
    visualization_type: str = Form(...),
    dataset_file: UploadFile = None
):
    try:
        if not dataset_file:
            raise HTTPException(status_code=400, detail="Dataset file is required.")

        # Load the dataset
        dataset = pd.read_csv(dataset_file.file)
        dataset_summary = dataset.describe(include="all").to_string()

        # Pass inputs to the LLM via LangChain
        response = chain.invoke({
            "features": features,
            "dataset_summary": dataset_summary,
            "visualization_type": visualization_type
        })

        # Clean the response
        cleaned_code = re.sub(r'^```python|```$', '', response, flags=re.MULTILINE)
        print(cleaned_code)

        # Prepare execution context
        exec_globals = {"plt": plt, "sns": sns, "pd": pd}
        exec_locals = {"dataset": dataset}
        exec(cleaned_code, exec_globals, exec_locals)

        # Save the plot to a BytesIO stream
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        # Return the image as a stream
        return StreamingResponse(buf, media_type="image/png")

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or not a valid CSV.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    


@app.post("/analyze-code/")
async def analyze_code(
    input_code : str = Form(...)
):
    try:
        # Clean the code input to remove invalid control characters
        # Use regex to replace single quotes with double quotes
        cleaned_code = re.sub(r"(?<!\\)'", '"', input_code)

        # Check if the cleaned code is empty or invalid
        if not cleaned_code.strip():
            raise HTTPException(status_code=400, detail="Code input cannot be empty or contain only invalid characters.")

        # Use json.loads to load the code into the required format
        try:
            loaded_code = json.loads(f'{{"code": {json.dumps(cleaned_code)}}}', strict=False)
        except json.JSONDecodeError as decode_error:
            raise HTTPException(status_code=400, detail=f"JSON Decode Error: {str(decode_error)}")

        # Pass the cleaned and loaded user input to the chain
        result = code_fix_chain.invoke(loaded_code)

        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/generate-case-study/")
async def generate_case_study(csv_file: UploadFile = File(...), concept: str = Form(...)):
    # Save the uploaded file temporarily
    file_path = f"temp_{csv_file.filename}"
    with open(file_path, "wb") as f:
        f.write(csv_file.file.read())
    
    try:
        # Generate the case study
        result = generate_case_study_v2(file_path, concept)
        return JSONResponse(content={"result": result})
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
