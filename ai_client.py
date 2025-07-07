from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import logging 
import torch
import google.generativeai as genai
import os

class AIClient:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)

        # Inicializar pipeline de geração de resposta com Gemini
        self._init_gemini_pipeline()
    
    def _init_gemini_pipeline(self):
        """Inicializa o pipeline de geração de resposta com Gemini"""
        try:
            # Configurar API do Gemini
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                self.logger.warning("⚠️ GEMINI_API_KEY não encontrada. Geração de resposta não estará disponível.")
                self.gemini_model = None
                return
            
            genai.configure(api_key=gemini_api_key)
            
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
            self.logger.info("✅ Pipeline de resposta Gemini configurado com sucesso")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao configurar Gemini: {e}")
            self.gemini_model = None
    
    
    def generate_response(self, text: str, classification, context: str = None, force_response: bool = False):
        """
        Gera uma resposta sugerida para o texto usando Gemini
        
        Args:
            text: Texto para o qual gerar resposta
            context: Contexto adicional (opcional)
            force_response: Se True, gera resposta mesmo para textos improdutivos
        """
        try:
            if self.gemini_model is None:
                return {
                    "success": False,
                    "message": "Geração de resposta não disponível. Verifique a configuração do Gemini.",
                    "response": None,
                    "classification": None
                }
                        
            # Se não forçar resposta e for improdutivo, não gera resposta
            if not force_response and classification.lower() == "improdutivo":
                return {
                    "success": True,
                    "message": "Mensagem classificada como improdutiva. Nenhuma resposta sugerida.",
                    "response": None,
                    "classification": classification
                }
            
            # Gera resposta usando Gemini
            response_text = self._generate_gemini_response(text, context, classification)
            
            return {
                "success": True,
                "message": "Resposta gerada com sucesso",
                "response": response_text,
                "classification": classification
            }
                
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {e}")
            return {
                "success": False,
                "message": f"Erro ao gerar resposta: {str(e)}",
                "response": None,
                "classification": None
            }
            
        
    def _generate_gemini_response(self, text: str, context: str = None, classification: str = None):
        """Gera resposta usando Gemini"""
        try:
            # Criar prompt personalizado baseado na classificação
            if classification and classification.lower() == "produtivo":
                system_instruction = """Você é um assistente profissional especializado em comunicação empresarial. 
                Gere uma resposta adequada, profissional e construtiva para mensagens de trabalho produtivas.
                Mantenha um tom cordial, objetivo e focado em resultados."""
            else:
                system_instruction = """Você é um assistente profissional. 
                Gere uma resposta educada e diplomática, tentando redirecionar a conversa para tópicos mais produtivos quando apropriado."""
            
            # Construir o prompt
            prompt_parts = [
                system_instruction,
                f"\nMensagem recebida: \"{text}\"",
            ]
            
            if context:
                prompt_parts.append(f"\nContexto adicional: {context}")
            
            if classification:
                prompt_parts.append(f"\nClassificação da mensagem: {classification}")
            
            prompt_parts.extend([
                "\nInstruções:",
                "- Gere uma resposta profissional e adequada",
                "- Mantenha um tom cordial e respeitoso",
                "- Seja conciso mas completo",
                "- Se necessário, sugira próximos passos ou ações",
                "- Responda em português brasileiro",
                "\nResposta sugerida:"
            ])
            
            prompt = "\n".join(prompt_parts)
            
            # Gerar resposta com Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Não foi possível gerar uma resposta adequada."
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta com Gemini: {e}")
            return f"Erro ao gerar resposta: {str(e)}"
        
    def _generate_gemini_classification(self, text: str, context: str = None, classification: str = None):
        """Gera resposta usando Gemini"""
        try:
            system_instruction = """Você é um assistente profissional. 
            Classifique a mensagem recebida em "produtivo" ou "improdutivo, considerando o contexto empresarial."""
            
            # Construir o prompt
            prompt_parts = [
                system_instruction,
                f"\nMensagem recebida: \"{text}\"",
            ]
            
            if context:
                prompt_parts.append(f"\nContexto adicional: {context}")
            
            if classification:
                prompt_parts.append(f"\nClassificação da mensagem: {classification}")
            
            prompt_parts.extend([
                "\nInstruções:",
                "- Gere uma resposta de apenas uma palavra",
                "- Considere que o texto foi recebido de um e-mail",
                "- Textos que tratam sobre reuniões empresariais, projetos, demandas, dentre outros, são considerados produtivos",
                "- Textos que tratam sobre correntes, festas, eventos pessoais, são considerados improdutivos",
                "- Responda em português brasileiro",
                "\nClassificação:"
            ])
            
            prompt = "\n".join(prompt_parts)
            
            # Gerar resposta com Gemini
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Não foi possível gerar uma resposta adequada."
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta com Gemini: {e}")
            return f"Erro ao gerar resposta: {str(e)}"
    
    def analyze_and_respond(self, text: str, context: str = None):
        """
        Método principal que combina classificação e geração de resposta
        
        Returns:
            dict: Resultado completo com classificação e resposta sugerida
        """
        try:
            # Classificar o texto
            classification = self._generate_gemini_classification(text)
            
            # Gerar resposta (apenas para textos produtivos por padrão)
            response_result = self.generate_response(text, classification, context)
            
            return {
                "text": text,
                "classification": {
                    "label": classification,
                    "is_productive": classification.lower() == "produtivo"
                },
                "response": response_result,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise completa: {e}")
            return {
                "text": text,
                "classification": {
                    "label": "Erro",
                    "is_productive": False
                },
                "response": {
                    "success": False,
                    "message": f"Erro na análise: {str(e)}",
                    "response": None,
                    "classification": None
                },
                "timestamp": self._get_timestamp()
            }
    
    def _get_timestamp(self):
        """Retorna timestamp atual"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_model_info(self):
        """Retorna informações sobre os modelos carregados"""
        return {
            "classification_model": {
                "path": self.classification_model_path,
                "loaded": self.classifier is not None
            },
            "response_model": {
                "type": "Gemini 1.5 Flash",
                "loaded": self.gemini_model is not None
            },
            "device": "GPU" if torch.cuda.is_available() else "CPU"
        }