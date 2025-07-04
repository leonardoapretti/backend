from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import logging 
import torch

class AIClient:
    def __init__(self, model_path="F:\\leonardo.pretti\\ai_models"):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        
        try:
            # Carrega o modelo e tokenizer locais
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Cria o pipeline de classificação
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1  # GPU se disponível
            )
            
            self.logger.info(f"Modelo carregado com sucesso de: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise
        
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
            
            # Adicione um log do que vai retornar
            if 'produtivo' == label:
                resultado_final = "Produtivo"
            elif 'improdutivo' == label:
                resultado_final = "Improdutivo"
            else:
                # ✅ FALLBACK: se não reconhecer a label, usa o score
                resultado_final = "Produtivo" if score > 0.5 else "Improdutivo"
            
            self.logger.info(f"Retornando para o front: {resultado_final}")
            return resultado_final
            
        except Exception as e:
            self.logger.error(f"Erro na classificação: {e}")
            return "Improdutivo"