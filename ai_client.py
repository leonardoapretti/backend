from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import logging 
import torch
import google.generativeai as genai
import os

class AIClient:
    def __init__(self, classification_model_path="F:\\leonardo.pretti\\ai_models"):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.classification_model_path = classification_model_path
        
        # Inicializar pipeline de classificação
        self._init_classification_pipeline()
        
        # Inicializar pipeline de geração de resposta com Gemini
        self._init_gemini_response_pipeline()
        
    def _init_classification_pipeline(self):
        """Inicializa o pipeline de classificação (seu modelo treinado)"""
        try:
            # Carrega o modelo e tokenizer locais
            self.tokenizer = AutoTokenizer.from_pretrained(self.classification_model_path)
            self.classification_model = AutoModelForSequenceClassification.from_pretrained(self.classification_model_path)
            
            # Cria o pipeline de classificação
            self.classifier = pipeline(
                "text-classification",
                model=self.classification_model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1  # GPU se disponível
            )
            
            self.logger.info(f"✅ Pipeline de classificação carregado de: {self.classification_model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo de classificação: {e}")
            raise
    
    def _init_gemini_response_pipeline(self):
        """Inicializa o pipeline de geração de resposta com Gemini"""
        try:
            # Configurar API do Gemini
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                self.logger.warning("⚠️ GEMINI_API_KEY não encontrada. Geração de resposta não estará disponível.")
                self.gemini_model = None
                return
            
            genai.configure(api_key=gemini_api_key)
            
            # Usar o modelo Gemini Flash (mais rápido e com maior cota gratuita)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
            self.logger.info("✅ Pipeline de resposta Gemini configurado com sucesso")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao configurar Gemini: {e}")
            self.gemini_model = None
        
    def generate_content(self, contents: str):
        """
        Usa o modelo local para classificar o conteúdo
        """
        try:
            # Faz a predição usando o pipeline
            result = self.classifier(contents)
            self.logger.info(f"Resultado da classificação: {result}")
            
            # Retorna o resultado com maior confiança
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao classificar conteúdo: {e}")
            return None
        
    def classify_productivity(self, text: str):
        """Classifica se o texto é produtivo ou improdutivo"""
        try:
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            self.logger.info(f"Classificando texto: {text[:100]}...")
            
            result = self.generate_content(text)
            self.logger.info("resultado: " + str(result))
            
            if result is None:
                return "Erro ao classificar"
            
            label = result.get('label', '').lower()
            score = result.get('score', 0)
            
            self.logger.info(f"Label: {label}, Score: {score}")
            
            # Determina o resultado final
            if 'produtivo' == label:
                resultado_final = "Produtivo"
            elif 'improdutivo' == label:
                resultado_final = "Improdutivo"
            else:
                # Fallback: se não reconhecer a label, usa o score
                resultado_final = "Produtivo" if score > 0.5 else "Improdutivo"
            
            self.logger.info(f"Retornando para o front: {resultado_final}")
            return resultado_final
            
        except Exception as e:
            self.logger.error(f"Erro na classificação: {e}")
            return "Improdutivo"
    
    def generate_response(self, text: str, context: str = None, force_response: bool = False):
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
            
            # Primeiro, classifica se é produtivo
            classification = self.classify_productivity(text)
            
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
    
    def analyze_and_respond(self, text: str, context: str = None):
        """
        Método principal que combina classificação e geração de resposta
        
        Returns:
            dict: Resultado completo com classificação e resposta sugerida
        """
        try:
            # Classificar o texto
            classification = self.classify_productivity(text)
            
            # Gerar resposta (apenas para textos produtivos por padrão)
            response_result = self.generate_response(text, context)
            
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

# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Inicializar o cliente AI
        ai_client = AIClient()
        
        # Verificar se os modelos foram carregados
        model_info = ai_client.get_model_info()
        print("Informações dos modelos:", model_info)
        
        # Exemplos de teste
        exemplos = [
            "Vamos marcar uma reunião para discutir o projeto amanhã?",
            "Preciso de ajuda com a implementação da nova funcionalidade",
            "Oi, tudo bem? Vamos tomar um café?",
            "O projeto foi entregue com sucesso, parabéns à equipe!",
            "Minhas compras chegaram hoje"
        ]
        
        print("\n" + "="*80)
        print("TESTANDO CLASSIFICAÇÃO E GERAÇÃO DE RESPOSTA")
        print("="*80)
        
        for i, exemplo in enumerate(exemplos, 1):
            print(f"\n{i}. Texto: '{exemplo}'")
            print("-" * 60)
            
            # Análise completa
            resultado = ai_client.analyze_and_respond(exemplo)
            
            # Exibir classificação
            classification = resultado['classification']
            emoji = "✅" if classification['is_productive'] else "❌"
            print(f"   Classificação: {emoji} {classification['label']}")
            
            # Exibir resposta
            response_data = resultado['response']
            if response_data['success'] and response_data['response']:
                print(f"   Resposta sugerida: '{response_data['response']}'")
            else:
                print(f"   {response_data['message']}")
            
            print("-" * 60)
        
    except Exception as e:
        print(f"❌ Erro ao executar exemplo: {e}")