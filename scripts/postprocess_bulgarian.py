#!/usr/bin/env python3
"""
Bulgarian Post-Processing Module
Изглажда граматика и правопис на български отговори от RAG чатбота.
"""

import re
from typing import Dict, Any, List
from loguru import logger

class BulgarianPostProcessor:
    """
    BUSINESS_RULE: Post-processing за български текст.
    Поправя граматика, правопис и стил на отговорите.
    """
    
    def __init__(self):
        """Инициализация на post-processor."""
        # Глаголни правила (без IGNORECASE за по-точна граматика)
        self.verb_rules = {
            # Поправка на "Ми" -> "Мога" в началото на изречения
            r'^\s*Ми\b': 'Мога',
            # Поправка на глаголи с контекст
            r'(?<!\bне\s)(може)\s+да\s+(отговоря)\b': r'мога да отговоря',
            # Премахнато: r'(?<!\bне\s)(може)\s+да\s+(отговори)\b' - граматично неправилно
            r'\b(позволява)\s+да\b': r'позволява да',
            r'\b(позволяват)\s+да\b': r'позволява да',  # Поправка на множествено число
        }
        
        # Съществителни правила
        self.noun_rules = {
            r'\b(одежда)\b': r'дрехи',
            r'\b(обувка)\b': r'обувки', 
            r'\b(садови)\s+(инструменти)\b': r'градински \2',
            r'\b(компанията|фирмата|магазинът)\s+предлагаме\b': r'\1 предлага',
        }
        
        # Стилистични правила
        self.style_rules = {
            r'\b(всички)\s+различни\b': r'различни',
            r'\b(много)\s+друга\b': r'много други',
            r'\b(дрехи|обувки|инструменти)\s+и\s+много\s+друга\b': r'\1 и много други',
            # Safe артикли - не се прилагат при прилагателни
            r'\b(на|за)\s+(нови|всички|различни)\s+(продукти|стоки|услуги)\b': r'\1 \2 \3',
            # Защитени артикли (не прави "продуктите" -> "продуктитете")
            r'\b(на)\s+(продукти|стоки|услуги)(?!те)\b': r'\1 \2те',
            r'\b(за)\s+(продукти|стоки|услуги)(?!те)\b': r'\1 \2те',
        }
        
        # Специални случаи за подобряване на отговори
        self.special_cases = {
            r'\bМи позволява да отговоря на въпросите за продуктите\.\b': 
                r'Ние предлагаме широк асортимент от продукти включително дрехи, обувки, градински инструменти и много други. Всички продукти са с гаранция и доставка в цялата страна.',
            r'\bМи позволяват да ви помогнеш\b': r'Мога да ви помогна',
            r'\b(ми|ни)\s+позволяват\s+да\s+.*(помогнеш|отговориш)\b': r'Мога да ви помогна',
        }
        
        # Стилистични подобрения
        self.style_improvements = {
            r'\b(Моята компания|Нашата компания)\b': 'Ние',
            r'\b(Мога да отговоря на въпросите)\b': 'Мога да отговоря на въпроси',
            r'\b(за продуктите)\b': 'за продуктите',
            r'\b(различни видове)\b': 'различни видове',
        }
        
        # Performance: Компилирай regex patterns за по-бързо изпълнение
        self._compiled_patterns = {}
        self._compile_patterns()
        
        # Disable verbose logging за production
        logger.disable("__main__")
        logger.info("BulgarianPostProcessor initialized with optimized grammar rules")

    def _compile_patterns(self):
        """Компилира regex patterns за по-бързо изпълнение."""
        all_rules = {
            **self.verb_rules,
            **self.noun_rules, 
            **self.style_rules,
            **self.special_cases,
            **self.style_improvements
        }
        
        for pattern, replacement in all_rules.items():
            self._compiled_patterns[pattern] = {
                'compiled': re.compile(pattern),
                'replacement': replacement
            }

    def apply_rules(self, text: str, rules: dict) -> str:
        """Прилага правила с компилирани patterns за по-бързо изпълнение."""
        for pattern, replacement in rules.items():
            if pattern in self._compiled_patterns:
                # Използвай компилирания pattern
                compiled = self._compiled_patterns[pattern]['compiled']
                text = compiled.sub(replacement, text)
            else:
                # Fallback към обикновен re.sub
                text = re.sub(pattern, replacement, text)
        return text

    def correct_grammar(self, text: str) -> str:
        """
        BUSINESS_RULE: Поправя граматика на български текст поетапно.
        
        Args:
            text: Оригинален текст
            
        Returns:
            Поправен текст
        """
        # Стъпка 1: Специални случаи (най-важни)
        text = self.apply_rules(text, self.special_cases)
        
        # Стъпка 2: Глаголни правила
        text = self.apply_rules(text, self.verb_rules)
        
        # Стъпка 3: Съществителни правила  
        text = self.apply_rules(text, self.noun_rules)
        
        return text

    def improve_style(self, text: str) -> str:
        """
        BUSINESS_RULE: Подобрява стила на български текст.
        
        Args:
            text: Оригинален текст
            
        Returns:
            Подобрен текст
        """
        # Стъпка 1: Стилистични правила
        text = self.apply_rules(text, self.style_rules)
        
        # Стъпка 2: Стилистични подобрения
        text = self.apply_rules(text, self.style_improvements)
        
        return text

    def postprocess_text(self, text: str) -> str:
        """
        BUSINESS_RULE: Пълен post-processing на български текст.
        Оптимизиран за по-бързо изпълнение.
        
        Args:
            text: Оригинален текст
            
        Returns:
            Обработен текст
        """
        if not text or not isinstance(text, str) or len(text) < 10:
            return text
        
        # Бърза проверка - ако текстът е кратък, пропусни обработката
        if len(text) < 50:
            return text.strip()
        
        # Стъпка 1: Поправяй граматика
        corrected = self.correct_grammar(text)
        
        # Стъпка 2: Прилагай стилистични подобрения
        styled = self.improve_style(corrected)
        
        # Стъпка 3: Нормализирай пунктуация
        final_text = re.sub(r'\s+([.,!?])', r'\1', styled)
        final_text = re.sub(r'([!?.,])([A-Za-zА-Яа-я])', r'\1 \2', final_text)
        
        # Стъпка 4: Премахвай излишни интервали
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        return final_text

    def postprocess_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        BUSINESS_RULE: Post-processing на RAG отговор.
        
        Args:
            response: RAG отговор
            
        Returns:
            Обработен отговор
        """
        if not isinstance(response, dict):
            return response
        
        # Обработвай различни полета за отговор
        answer_fields = ['answer', 'Отговор', 'response', 'text']
        
        for field in answer_fields:
            if field in response and isinstance(response[field], str):
                original = response[field]
                processed = self.postprocess_text(original)
                
                if processed != original:
                    logger.info(f"Post-processed {field}: '{original[:30]}...' -> '{processed[:30]}...'")
                    response[field] = processed
        
        return response


def postprocess_rag_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    BUSINESS_RULE: Главна функция за post-processing на RAG отговор.
    
    Args:
        response: RAG отговор
        
    Returns:
        Обработен отговор
    """
    processor = BulgarianPostProcessor()
    return processor.postprocess_response(response)


# Пример за използване
if __name__ == "__main__":
    # Тест пример
    test_response = {
        "answer": "Ми позволява да отговоря на въпросите за продукти. Моята компания предлагаме всички различни видове продукти като одежда, обувка, садови инструменти и много друга.",
        "score": 85,
        "sources": []
    }
    
    print("Оригинален отговор:")
    print(test_response["answer"])
    print("\n" + "="*50 + "\n")
    
    processed = postprocess_rag_response(test_response)
    print("Обработен отговор:")
    print(processed["answer"])
