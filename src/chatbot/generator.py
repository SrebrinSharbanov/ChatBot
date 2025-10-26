"""
LLM Generator wrapper for answer generation.
Supports Ollama, OpenAI, and other LLM providers.
"""

from typing import List, Dict, Any, Optional
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # pyright: ignore[reportMissingImports]

from config.settings import settings


class LLMGenerator:
    """
    BUSINESS_RULE: LLM wrapper for generating answers from context.
    Supports multiple LLM providers (Ollama, OpenAI, etc.)
    """
    
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        ollama_host: str = None
    ):
        """
        Initialize LLM generator.
        
        Args:
            provider: LLM provider ('ollama', 'openai', 'local')
            model_name: Model name/identifier
            ollama_host: Ollama API host (if using Ollama)
        """
        self.provider = provider or settings.llm.provider
        self.model_name = model_name or settings.llm.model_name
        self.ollama_host = ollama_host or settings.llm.ollama_host
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens
        self.timeout = settings.llm.timeout
        
        logger.info(f"LLM Generator initialized: {self.provider}/{self.model_name}")
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        BUSINESS_RULE: Build prompt for LLM with RAG context.
        Instructs model to answer ONLY from provided context.
        
        Args:
            query: User query
            context: Retrieved context from documents
            sources: List of source citations
        
        Returns:
            Formatted prompt string
        """
        # Format sources for citation
        source_list = ", ".join([
            f"{s['source_table']}:{s['source_id']}"
            for s in sources[:3]  # Limit to top 3
        ])
        
        prompt = f"""Ти си полезен асистент който отговаря на въпроси на български език.

ВАЖНО: Използвай САМО информацията от контекста по-долу за да отговориш на въпроса. 
НЕ измисляй или добавяй информация която не е в контекста.

КОНТЕКСТ:
{context}

ВЪПРОС: {query}

ОТГОВОР (на български, полезен и информативен, 3-4 изречения с конкретни детайли):"""

        return prompt
    
    @retry(
        stop=stop_after_attempt(3),  # Back to original
        wait=wait_exponential(multiplier=2, min=4, max=30),  # Back to original
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
    )
    def _call_ollama(self, prompt: str) -> str:
        """
        BUSINESS_RULE: Call Ollama API for generation.
        Includes retry logic for resilience.
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            Generated answer text
        """
        url = f"{self.ollama_host}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        logger.debug(f"Calling Ollama API: {url}")
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip()
            
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _call_openai(self, prompt: str) -> str:
        """
        BUSINESS_RULE: Call OpenAI API for generation (alternative provider).
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            Generated answer text
        """
        try:
            import openai  # pyright: ignore[reportMissingImports]
            openai.api_key = settings.llm.get("openai_api_key", "")
            
            response = openai.ChatCompletion.create(
                model=self.model_name or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        BUSINESS_RULE: Generate answer using LLM based on retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            sources: Source citations
        
        Returns:
            Generated answer text
        """
        # Build prompt
        prompt = self._build_prompt(query, context, sources)
        
        # Call appropriate LLM provider
        if self.provider == "ollama":
            answer = self._call_ollama(prompt)
        elif self.provider == "openai":
            answer = self._call_openai(prompt)
        else:
            # Fallback: extractive answer (first 400 chars of top result)
            logger.warning(f"Unknown provider '{self.provider}', using extractive fallback")
            answer = context[:400] if context else "Не мога да намеря отговор."
        
        return answer
    
    def generate_simple_answer(self, context: str, max_length: int = 400) -> str:
        """
        BUSINESS_RULE: Generate simple extractive answer (fallback).
        Returns first part of context as answer.
        
        Args:
            context: Context text
            max_length: Maximum answer length
        
        Returns:
            Simple extracted answer
        """
        if not context:
            return "Не мога да намеря релевантна информация."
        
        # Extract first segment/paragraph
        parts = context.split("\n\n---\n\n")
        if parts:
            first_part = parts[0]
            # Remove source label
            if "]" in first_part:
                first_part = first_part.split("]", 1)[1].strip()
            return first_part[:max_length]
        
        return context[:max_length]
    
    def generate_from_prompt(self, prompt: str) -> str:
        """
        BUSINESS_RULE: Generate answer from a custom prompt (for product filtering, etc).
        Useful for generating formatted responses from structured data.
        
        PERFORMANCE_CRITICAL: Direct LLM call with custom prompt,
        avoiding extra prompt building overhead.
        
        Args:
            prompt: Custom formatted prompt for LLM
        
        Returns:
            Generated answer text
        """
        logger.debug(f"Generating from custom prompt (provider: {self.provider})")
        
        try:
            # Call appropriate LLM provider
            if self.provider == "ollama":
                answer = self._call_ollama(prompt)
            elif self.provider == "openai":
                answer = self._call_openai(prompt)
            else:
                # Fallback for unknown provider
                logger.warning(f"Unknown provider '{self.provider}', returning empty")
                answer = ""
            
            if not answer:
                logger.warning("LLM returned empty response")
                return ""
            
            # Don't filter by length - let caller handle short responses
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating from prompt: {e}")
            raise

    @staticmethod
    def fix_bulgarian_grammar(text: str) -> str:
        """
        BUSINESS_RULE: Post-process Bulgarian text to fix literal translations and unnatural phrasing.
        
        Args:
            text: Raw text from LLM or retrieval
            
        Returns:
            Polished Bulgarian text
        """
        replacements = {
            "Моля, представете": "Предлагаме",
            "Виждаме, че": "Разполагаме с",
            "Виждаме възможността": "Можем да предложим",
            "Има ли повърхност": "Предлагаме",
            "пълно възстановяване": "пълно възстановяване на сумата",
            "не откривам продукти": "в момента нямаме такива продукти",
            "не мога да намеря": "не откривам в нашия каталог",
            "налични продукти": "налични продукти в нашия каталог",
            "различните видове устройства": "различни видове устройства",
            "Всички наши продукти се обично": "Всички наши продукти обикновено се",
        }
        
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        logger.debug(f"Bulgarian grammar fixed: {len(replacements)} replacements applied")
        return result

