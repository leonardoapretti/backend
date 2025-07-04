from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from ai_client import AIClient
import logging
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# from ai_client import AIClient

class EmailClassifierAPI:
    def __init__(self, model_path: str, allowed_origins=None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.allowed_origins = allowed_origins or ["http://localhost:5173"]
        self.app = FastAPI()
        self._setup_middleware()
        self.ai_client = AIClient()
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
            email_file: UploadFile = File(None)
        ):
            if not email_text and email_file:
                content = await email_file.read()
                
                # Primeira tentativa: chardet
                try:
                    import chardet
                    detected = chardet.detect(content)
                    encoding = detected['encoding'] or 'utf-8'
                    self.logger.info(f"Encoding detectado: {encoding}")
                    email_text = content.decode(encoding)
                except (ImportError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Falha com chardet: {e}")
                    # Fallback: tenta encodings comuns
                    email_text = None
                    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            email_text = content.decode(encoding)
                            self.logger.info(f"Sucesso com encoding: {encoding}")
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if email_text is None:
                        # Última tentativa: decode com errors='replace'
                        try:
                            email_text = content.decode('utf-8', errors='replace')
                            self.logger.warning("Usando decode com errors='replace'")
                        except Exception as e:
                            return {"error": f"Não foi possível decodificar o arquivo: {str(e)}"}
                            
            elif not email_text:
                return {"error": "Nenhum texto fornecido"}
            
            classification = self.ai_client.classify_productivity(email_text)
            
            return {
                "response": email_text,
                "category": classification
            }

# Instancie a API
email_api = EmailClassifierAPI(model_path="../ai/modelo_treinado")

# Para rodar com uvicorn:
app = email_api.app