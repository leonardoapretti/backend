from google import genai

from dotenv import load_dotenv
import logging 

class AIClient:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.client = genai.Client()
        self.productivity_prompt = ("""
        Classifique o e-mail como "produtivo" ou "improdutivo" conforme as definições abaixo. 
        Responda apenas com a classificação, sem explicações.
        produtivo: e-mails sobre trabalho, tarefas, projetos, reuniões, calls, stack, decisões ou informações úteis.
        improdutivo: e-mails irrelevantes, pessoais, correntes, café, promoções ou spam.
        Se não tiver certeza, classifique como "improdutivo".
        E-mail: {email_text}
        Classificação:
        """)
        
    def generate_content(self, contents: str, model: str = "gemini-2.5-flash"):
        try:
            response = self.client.models.generate_content(
                model=model, contents=contents
            )
            # Logue a resposta bruta para depuração
            self.logger.info(f"Resposta bruta da API: {response}")
            # Tente acessar o texto de diferentes formas
            if hasattr(response, "text") and response.text:
                return response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Gemini pode retornar uma lista de candidatos
                return response.candidates[0].text
            else:
                return None
        except Exception as e:
            self.logger.error(f"Erro ao gerar conteúdo: {e}")
            return None
        
    def classify_productivity(self, text: str, model: str = "gemini-2.5-flash"):
        prompt = self.productivity_prompt.format(email_text=text)
        self.logger.info(prompt)
        response = self.generate_content(prompt)
        self.logger.info(response)
        
        cleaned_response = response.strip().lower()
        if "produtivo" in cleaned_response and "improdutivo" not in cleaned_response:
            return "Produtivo"
        elif "improdutivo" in cleaned_response:
            return "Improdutivo"
        else:
            return "Improdutivo"
