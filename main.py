from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import logging

from ai_client import AIClient

class EmailClassifierAPI:
    def __init__(self, model_path: str):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path

        self.allowed_origins = [
            "http://localhost:5173",
            "https://email-classifier-frontend-eta.vercel.app",
            "https://email-classifier-frontend-e2mwlemm7-leonardoaprettis-projects.vercel.app",
        ]

        self.app = FastAPI(
            title="AutoU AI API",
            description="API para classificação de produtividade e geração de respostas",
            version="1.0.0"
        )
        self._setup_middleware()

        try:
            self.ai_client = AIClient()
            self.logger.info("✅ AI Client inicializado com sucesso")
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar AI Client: {e}")
            self.ai_client = None

        self._setup_routes()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        @self.app.post("/api/process_email/")
        async def process_email(
            email_text: str = Form(None),
            email_file: UploadFile = File(None),
            context: Optional[str] = Form(None)
        ):
            if not self.ai_client:
                raise HTTPException(status_code=503, detail="AI Client não inicializado")

            if not email_text and email_file:
                content = await email_file.read()
                try:
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected['encoding'] or 'utf-8'
                    self.logger.info(f"Encoding detectado: {encoding}")
                    email_text = content.decode(encoding)
                except (ImportError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Falha com chardet: {e}")
                    email_text = None
                    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            email_text = content.decode(encoding)
                            self.logger.info(f"Sucesso com encoding: {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue

                    if email_text is None:
                        try:
                            email_text = content.decode('utf-8', errors='replace')
                            self.logger.warning("Usando decode com errors='replace'")
                        except Exception as e:
                            raise HTTPException(status_code=400, detail=f"Não foi possível decodificar o arquivo: {str(e)}")

            elif not email_text:
                raise HTTPException(status_code=400, detail="Nenhum texto ou arquivo fornecido")

            try:
                self.logger.info(f"Processando email: {email_text[:100]}...")
                result = self.ai_client.analyze_and_respond(email_text, context)
                classification = result['classification']
                response_data = result['response']

                return {
                    "success": True,
                    "text": email_text,
                    "classification": {
                        "category": classification['label'],
                        "is_productive": classification['is_productive']
                    },
                    "response": {
                        "generated": response_data['success'],
                        "message": response_data['message'],
                        "text": response_data['response']
                    },
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Erro ao processar email: {e}")
                raise HTTPException(status_code=500, detail=f"Erro ao processar email: {str(e)}")

# Instanciar a API
email_api = EmailClassifierAPI(model_path="ai_model")

app = email_api.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
