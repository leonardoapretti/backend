from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/process_email/")
async def process_email(
    email_text: str = Form(None),
    email_file: UploadFile = File(None)
):
   
    if not email_text and email_file:
        content = await email_file.read()
        email_text = content.decode("utf-8")
    elif not email_text:
        return {"error": "Nenhum texto fornecido"}

    return {
        "response": email_text,
        'category':'Produtiva'
        }