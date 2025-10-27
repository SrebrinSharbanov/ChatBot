"""
RAG (Retrieval-Augmented Generation) Chatbot.
Main chatbot logic combining retrieval and generation.
"""

from typing import Dict, Any, Optional
from loguru import logger

from config.settings import settings
from src.retrieval.retriever import Retriever
from src.retrieval.product_filter import ProductFilter
from src.indexing.embeddings import EmbeddingModel
from src.database.db import get_db_connection
from .generator import LLMGenerator
from .intent_recognizer import IntentRecognizer
from scripts.postprocess_bulgarian import postprocess_rag_response


class RAGChatbot:
    """
    BUSINESS_RULE: Main RAG chatbot combining retrieval and generation.
    
    Flow:
    1. User asks question
    2. Retrieve top-k relevant segments
    3. Calculate confidence score
    4. If score >= threshold: Generate answer with LLM
    5. If score < threshold: Return "Това не е в моята компетенция"
    6. Return answer + sources + score
    """
    
    def __init__(self):
        """Initialize RAG chatbot with retriever, generator, and intent recognizer."""
        self.retriever = Retriever()
        self.generator = LLMGenerator()
        
        # Initialize intent recognizer with embedding model
        embedding_model = EmbeddingModel()
        self.intent_recognizer = IntentRecognizer(embedding_model)
        self.threshold = settings.rag.score_threshold
        
        logger.info(f"RAG Chatbot initialized (threshold: {self.threshold})")
    
    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        BUSINESS_RULE: Main chatbot query interface.
        
        Process:
        1. Retrieve relevant documents
        2. Calculate score
        3. If score >= threshold: generate answer
        4. Else: return decline message
        5. Log query for analytics
        
        Args:
            query: User question
            top_k: Number of documents to retrieve (default from config)
            threshold: Score threshold (default from config)
        
        Returns:
            Dictionary with answer, score, and sources
        """
        threshold = threshold or self.threshold
        
        logger.info(f"Processing query: '{query}'")
        
        # Step 1: Intent recognition and query processing
        intent_result = self.intent_recognizer.process_query(query)
        processed_query = intent_result['processed_query']
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f}), Processed: '{processed_query}'")
        
        # DEBUG: Print intent recognition details
        print(f"[DEBUG] Recognized intent = '{intent}' | confidence = {confidence:.2f} | processed_query = '{processed_query}'")
        print(f"[DEBUG] Intent type: {type(intent)}, Intent repr: {repr(intent)}")
        
        # Step 2: Special routing for products intent
        if intent == "products":
            print(f"[DEBUG] Routing to products handler")
            return self._handle_products_query(query, intent_result)
        else:
            print(f"[DEBUG] Intent '{intent}' != 'products', using normal retrieval")
        
        # Step 3: Retrieve relevant documents using processed query
        retrieval_result = self.retriever.retrieve(processed_query, top_k=top_k)
        
        score = retrieval_result['score']
        should_answer = retrieval_result['should_answer']
        sources = retrieval_result['sources']
        context = retrieval_result['context']
        
        logger.info(f"Retrieval score: {score}/100 (threshold: {threshold})")
        
        # Step 2: Decide whether to answer or decline
        if not should_answer or score < threshold:
            # BUSINESS_RULE: Products intent fallback - use product list when score is low
            if intent == "products":
                logger.warning(f"Low score ({score}) for products - using fallback product list")
                return self._fallback_product_list(query, intent, score)
            
            # BUSINESS_RULE: Low confidence → decline to answer
            answer = "Това не е в моята компетенция."
            logger.info("Score below threshold - declining to answer")
            
            return {
                "query": query,
                "answer": answer,
                "score": score,
                "sources": [],
                "confidence": "low"
            }
        
        # Step 3: Generate answer using LLM
        try:
            logger.info("Generating answer with LLM...")
            answer = self.generator.generate(
                query=query,
                context=context,
                sources=sources
            )
            
            # Fallback if LLM fails or returns empty
            if not answer or len(answer.strip()) < 10:
                logger.warning("LLM returned empty/short answer, using extractive fallback")
                answer = self.generator.generate_simple_answer(context)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to extractive answer
            answer = self.generator.generate_simple_answer(context)
        
        # Step 4: Format response
        response = {
            "query": query,
            "answer": answer,
            "score": score,
            "sources": sources,
            "confidence": "high" if score >= 90 else "medium",
            "intent": intent,
            "intent_confidence": confidence,
            "processed_query": processed_query
        }
        
        # Step 5: Post-process Bulgarian text
        response = postprocess_rag_response(response)
        
        logger.success(f"Generated answer ({len(answer)} chars, {len(sources)} sources)")
        
        return response

    def _handle_product_not_found(self, query: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        BUSINESS_RULE: Handle cases where user asks for a product that doesn't exist in database.
        
        Args:
            query: User's original query
            intent_result: Intent recognition result
            
        Returns:
            Response indicating product is not available
        """
        try:
            product_filter = intent_result.get('product_filter', 'този продукт')
            
            # Create a helpful response message
            answer = f"Към момента не предлагаме {product_filter}. "
            answer += "Можете да разгледате нашия каталог с налични продукти или да се свържете с нас за повече информация."
            
            logger.info(f"Product not found response for '{query}' (product: {product_filter})")
            
            return {
                "query": query,
                "answer": answer,
                "score": 90,  # High score because we understood the intent correctly
                "sources": [],
                "confidence": "high",
                "intent": "product_not_found",
                "intent_confidence": intent_result['confidence'],
                "processed_query": intent_result['processed_query'],
                "product_filter": product_filter
            }
            
        except Exception as e:
            logger.error(f"Error handling product not found: {e}")
            return {
                "query": query,
                "answer": "Това не е в моята компетенция.",
                "score": 0,
                "sources": [],
                "confidence": "low",
                "intent": "product_not_found",
                "intent_confidence": 0.5,
                "processed_query": query
            }

    def _handle_products_query(self, query: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        BUSINESS_RULE: Intelligent products handler with hybrid retrieval.
        
        Process:
        1. Extract product keywords from user query
        2. Try keyword-based SQL filtering first (fast, precise)
        3. If insufficient results, fall back to vector search (comprehensive)
        4. Format products for LLM
        5. Generate natural language response with LLM
        6. Return response with sources and high score
        """
        try:
            logger.info(f"Handling products query: '{query}'")
            
            # Step 1: Extract product keywords from query
            keywords = ProductFilter.extract_keywords(query)
            logger.info(f"Extracted product keywords: {keywords}")
            
            # Вземи връзка с базата данни
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Step 2: Try keyword-based SQL filtering FIRST
            keyword_results = []
            if keywords:
                # BUSINESS_RULE: Специална логика за общи продуктови заявки
                # Приоритизираме конкретните категории над общия "продукти"
                specific_keywords = [kw for kw in keywords if kw != "продукти"]
                
                if specific_keywords:
                    # За конкретни категории - използваме филтриране
                    where_clause = ProductFilter.build_filter_query(specific_keywords)
                    sql = f"""
                        SELECT name, description, category, price 
                        FROM products 
                        {where_clause}
                        ORDER BY price ASC
                        LIMIT 10
                    """
                    logger.info(f"Executing keyword-based SQL filter for: {specific_keywords}")
                elif "продукти" in keywords:
                    # За "какви продукти имате" - показваме всички продукти
                    sql = """
                        SELECT name, description, category, price 
                        FROM products 
                        ORDER BY id 
                        LIMIT 10
                    """
                    logger.info("General products query - fetching all products")
                
                cursor.execute(sql)
                keyword_results = cursor.fetchall()
                logger.info(f"Keyword search found {len(keyword_results)} products")
            
            # Step 3: Decide if we need fallback to vector search
            products = keyword_results
            search_method = "keyword"
            response_score = 95
            
            # BUSINESS_RULE: Important categories never fallback if at least 1 result exists
            should_fallback = ProductFilter.should_use_fallback(keyword_results, min_results=2, keywords=keywords)
            
            if should_fallback:
                logger.info(f"Keyword search insufficient (found {len(keyword_results)}) - falling back to vector search")
                cursor.close()
                conn.close()
                
                # Fallback to vector/retriever search
                retrieval_result = self.retriever.retrieve(query, top_k=10)
                vector_results = retrieval_result['results']
                
                # Merge keyword and vector results
                products = ProductFilter.merge_results(keyword_results, vector_results)
                search_method = "hybrid" if keyword_results else "vector"
                response_score = 85 if keyword_results else 75
            else:
                logger.info(f"Keyword search sufficient ({len(keyword_results)} products found) - no fallback")
                cursor.close()
                conn.close()
            
            # Step 4: Check if products found
            if not products:
                # Extract product name from query for better message
                product_filter = intent_result.get('product_filter', None)
                if product_filter:
                    no_products_message = f"Към момента не предлагаме {product_filter}."
                else:
                    no_products_message = "В момента не откривам продукти от този тип."
                
                if keywords:
                    no_products_message += f" Търсихме: {', '.join(keywords)}"
                
                no_products_message += " Можете да разгледате нашия каталог с налични продукти или да се свържете с нас за повече информация."
                
                logger.warning(f"No products found for keywords: {keywords}")
                
                cursor.close()
                conn.close()
                
                return {
                    "query": query,
                    "answer": no_products_message,
                    "score": 90,  # High score because we understood the intent correctly
                    "sources": [],
                    "confidence": "high",
                    "intent": "products",
                    "intent_confidence": intent_result['confidence'],
                    "processed_query": intent_result['processed_query']
                }
            
            # Step 5: Format products for LLM
            is_filtered = len(keywords) > 0 and search_method == "keyword"
            products_text = ProductFilter.format_products_for_llm(products, include_category=(search_method == "keyword"))
            
            # Step 6: Create LLM prompt and generate response
            llm_prompt = ProductFilter.create_llm_prompt(query, products_text, is_filtered=is_filtered)
            
            logger.info(f"Generating LLM response for {len(products)} products (method: {search_method}, score: {response_score})")
            try:
                # PERFORMANCE_CRITICAL: Call LLM to generate natural response
                answer = self.generator.generate_from_prompt(llm_prompt)
                
                # Fallback if LLM returns empty
                if not answer or len(answer.strip()) < 10:
                    logger.warning("LLM returned empty/short response - using formatted list")
                    answer = f"В нашия каталог предлагаме:\n\n{products_text}"
                
                # Post-process Bulgarian grammar for natural phrasing
                answer = LLMGenerator.fix_bulgarian_grammar(answer)
                
            except Exception as e:
                logger.error(f"LLM generation failed: {e} - using formatted list")
                answer = f"В нашия каталог предлагаме:\n\n{products_text}"
                answer = LLMGenerator.fix_bulgarian_grammar(answer)
            
            # Step 7: Create sources list
            sources = []
            for i, (name, description, category, price) in enumerate(products):
                sources.append({
                    "source_id": f"product_{i+1}",
                    "source_table": "products",
                    "segment_id": f"products:{i+1}:seg0",
                    "title": name,
                    "similarity": 1.0,
                    "category": category,
                    "price": price if price > 0 else None
                })
            
            # Step 8: Build response with adaptive score
            response = {
                "query": query,
                "answer": answer,
                "score": response_score,  # Адаптивен score (95 за keyword, 75 за vector)
                "sources": sources,
                "confidence": "high" if response_score >= 85 else "medium",
                "intent": "products",
                "intent_confidence": intent_result['confidence'],
                "processed_query": intent_result['processed_query'],
                "filter_keywords": keywords,
                "search_method": search_method  # За debugging
            }
            
            cursor.close()
            conn.close()
            
            # Post-process Bulgarian text
            response = postprocess_rag_response(response)
            
            logger.success(f"Products query handled: {len(products)} products found, method={search_method}, keywords={keywords}")
            return response
            
        except Exception as e:
            logger.error(f"Error handling products query: {e}")
            logger.exception(e)
            # Fallback към обикновеното търсене при грешка
            return self._fallback_to_normal_search(query, intent_result)

    def _fallback_to_normal_search(self, query: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback към обикновеното търсене при грешка."""
        logger.warning("Falling back to normal search for products query")
        # Тук можеш да имплементираш fallback логика
        return {
            "query": query,
            "answer": "Извиняваме се, но в момента не можем да покажем продуктите. Моля опитайте отново.",
            "score": 0,
            "sources": [],
            "confidence": "low",
            "intent": "products",
            "intent_confidence": intent_result['confidence'],
            "processed_query": intent_result['processed_query']
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        VALIDATION: Check if chatbot components are healthy.
        
        Returns:
            Health status dictionary
        """
        status = {
            "retriever": "unknown",
            "generator": "unknown",
            "database": "unknown"
        }
        
        try:
            # Test retriever
            test_results = self.retriever.search("test", top_k=1)
            status["retriever"] = "ok" if test_results is not None else "error"
        except Exception as e:
            logger.error(f"Retriever health check failed: {e}")
            status["retriever"] = "error"
        
        try:
            # Test generator (simple check)
            if self.generator.provider == "ollama":
                import requests
                response = requests.get(f"{self.generator.ollama_host}/api/tags", timeout=5)
                status["generator"] = "ok" if response.ok else "error"
            else:
                status["generator"] = "ok"  # Assume ok for other providers
        except Exception as e:
            logger.error(f"Generator health check failed: {e}")
            status["generator"] = "error"
        
        # Overall status
        status["overall"] = "ok" if all(v == "ok" for k, v in status.items() if k != "overall") else "degraded"
        
        return status
    
    def _fallback_product_list(self, query: str, intent: str, score: int):
        """
        BUSINESS_RULE: Fallback for products queries when retrieval score is low.
        Returns a general product list when specific products aren't found.
        
        Args:
            query: Original user query
            intent: Recognized intent
            score: Retrieval score
            
        Returns:
            Dict with fallback answer
        """
        try:
            # Try to get products from database
            products = []
            conn = None
            try:
                from src.database.db import get_db_connection
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT name, category FROM products LIMIT 5")
                products = cursor.fetchall()
                cursor.close()
            except Exception as e:
                logger.error(f"Failed to fetch products: {e}")
            finally:
                if conn:
                    conn.close()
            
            if products:
                # Format product list
                product_names = [f"{name} ({category})" for name, category in products]
                joined = ", ".join(product_names)
                answer = f"Да, предлагаме различни продукти като: {joined}. Моля, формулирайте по-конкретен въпрос за по-точна информация."
            else:
                # Static fallback
                answer = "Да, предлагаме разнообразни продукти включително лаптопи, телефони, таблети, часовници и аксесоари. Моля, задайте по-конкретен въпрос."
            
            logger.info(f"Fallback product list returned")
            
            return {
                "query": query,
                "answer": answer,
                "score": score,
                "sources": [],
                "confidence": "low"
            }
        except Exception as e:
            logger.error(f"Fallback product list error: {e}")
            return {
                "query": query,
                "answer": "Предлагаме разнообразни продукти, моля посетете нашия каталог за повече информация.",
                "score": score,
                "sources": [],
                "confidence": "low"
            }

