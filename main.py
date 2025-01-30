from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import re
import json
from visualization import chain
from codefix import code_fix_chain

app = FastAPI()


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

        # Return the response
        return JSONResponse(content={"generated_code": cleaned_code})

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


