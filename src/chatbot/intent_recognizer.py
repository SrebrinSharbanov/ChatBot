"""
Intent Recognition and Query Expansion module.
Provides semantic understanding of user queries and intelligent query rewriting.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util
from loguru import logger
import re
import simplemma

from config.settings import settings


class IntentRecognizer:
    """
    BUSINESS_RULE: Semantic intent recognition and query expansion.
    Transforms vague user queries into precise search queries using embeddings.
    """
    
    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize intent recognizer.
        
        Args:
            embedding_model: Pre-loaded sentence transformer model
        """
        self.embedding_model = embedding_model
        
        # Initialize Bulgarian lemmatization (simplemma)
        try:
            # simplemma 1.1.2 doesn't have setup() method
            logger.info("Bulgarian lemmatization initialized (simplemma)")
        except Exception as e:
            logger.warning(f"Failed to initialize Bulgarian lemmatization: {e}")
        
        # Canonical queries for different intents
        self.canonical_queries = {
            "payment": [
                "Какви са начините на плащане?",
                "Методи за плащане",
                "Как мога да платя?",
                "Опции за плащане",
                "Плащане с карта",
                "Банков превод",
                "Начини на плащане",
                "Как да платя",
                "Плащане на поръчка"
            ],
            "delivery": [
                "Колко време отнема доставката?",
                "Време за доставка",
                "Кога ще пристигне поръчката?",
                "Срокове за доставка",
                "Експресна доставка",
                "Доставка до адрес"
            ],
            "return_policy": [
                "Как се връща продукт?",
                "Политика за връщане",
                "Рекламации",
                "Връщане на стоки",
                "Гаранция за връщане",
                "Условия за връщане"
            ],
            "warranty": [
                "Каква е гаранцията?",
                "Гаранция за продукти",
                "Сервиз и поддръжка",
                "Гаранционен срок",
                "Ремонт на продукти"
            ],
            "shipping": [
                "Доставка в чужбина",
                "Международна доставка",
                "Доставка в Европа",
                "Транспортни разходи",
                "Куриерска доставка"
            ],
            "contact": [
                "Как да се свържа с вас?",
                "Контактна информация",
                "Телефон за връзка",
                "Email за поддръжка",
                "Адрес на магазина"
            ],
            "working_hours": [
                "Работите ли в неделя?",
                "Кога работите?",
                "Работно време",
                "Отворени ли сте в събота?",
                "До колко часа работите?",
                "От колко часа отваряте?",
                "Работите ли в почивни дни?",
                "График на работа"
            ],
            "products": [
                "Какви продукти имате?",
                "Предлагате ли лаптопи?",
                "Имате ли телефони?",
                "Продавате ли часовници?",
                "Какво имате в каталога?",
                "Покажете ми продуктите",
                "Имате ли таблети?",
                "Предлагате ли компютри?",
                "Какви стоки имате?",
                "Каталог с продукти",
                "Налични стоки",
                "Нови продукти",
                "Популярни продукти"
            ],
            "promotions": [
                "Имате ли промоции?",
                "Какви са акциите?",
                "Имате ли промо код?",
                "Има ли отстъпки?",
                "Има ли намаления?",
                "Имате ли ваучери?",
                "Има ли специални оферти?",
                "Промоционални кодове",
                "Отстъпки за студенти",
                "Корпоративни отстъпки"
            ]
        }
        
        # Context boosters for semantic enhancement
        self.context_boosters = {
            "payment": "начини на плащане, плащане онлайн, наложен платеж, карта, банков превод",
            "delivery": "доставка, куриер, време за доставка, срок, доставка до адрес",
            "return_policy": "връщане на стоки, рекламации, гаранция, условия за връщане",
            "warranty": "гаранция, сервиз, ремонт, поддръжка, гаранционен срок",
            "shipping": "доставка, куриер, транспорт, разходи за доставка",
            "contact": "телефон, email, адрес, връзка, поддръжка",
            "products": "каталог, продукти, стоки, асортимент, налични",
            "promotions": "промоции, акции, отстъпки, намаления, промо код, ваучери, оферти",
            "working_hours": "работно време, график, понеделник, вторник, сряда, четвъртък, петък, събота, неделя, почивни дни"
        }
        
        # Canonical aliases for query normalization (expanded with conversational forms)
        self.aliases = {
            # Payment aliases
            "как се плаща": "начини на плащане",
            "как да платя": "начини на плащане",
            "как да плащам": "начини на плащане",
            "начини на плащане": "начини на плащане", 
            "плащане": "начини на плащане",
            "плаща": "начини на плащане",
            "плащам": "начини на плащане",
            "разплащане": "начини на плащане",
            "приемате ли карта": "начини на плащане",
            "плащане с карта": "начини на плащане",
            "банков превод": "начини на плащане",
            
            # Delivery aliases
            "доставка": "време за доставка",
            "кога пристига": "време за доставка",
            "кога ще дойде": "време за доставка",
            "кога ще пристигне": "време за доставка",
            "срок за доставка": "време за доставка",
            "време за доставка": "време за доставка",
            "колко време отнема": "време за доставка",
            "кога ще дойде пратката": "време за доставка",
            
            # Return aliases
            "връщане": "политика за връщане",
            "върна": "политика за връщане",
            "как да върна": "политика за връщане",
            "искам да върна": "политика за връщане",
            "рекламация": "политика за връщане",
            
            # Products aliases
            "какви продукти имате": "каталог продукти",
            "какво продавате": "каталог продукти", 
            "имате ли продукти": "каталог продукти",
            "предлагате ли": "каталог продукти",
            "имате ли стоки": "каталог продукти",
            "каталог": "каталог продукти",
            "продукти": "каталог продукти",
            "стоки": "каталог продукти",
            "асаортимент": "каталог продукти",
            "има ли": "каталог продукти",
            "наличност": "каталог продукти",
            "наличен": "каталог продукти",
            "в наличност": "каталог продукти",
            "купя": "каталог продукти",
            "поръчам": "каталог продукти",
            "вземете ли": "каталог продукти",
            "политика за връщане": "политика за връщане",
            "върна продукт": "политика за връщане",
            "върна стока": "политика за връщане",
            
            # Warranty aliases
            "гаранция": "гаранция за продукти",
            "има ли гаранция": "гаранция за продукти",
            "гаранция на стоката": "гаранция за продукти",
            "сервиз": "гаранция за продукти",
            "ремонт": "гаранция за продукти",
            "гаранция за продукти": "гаранция за продукти",
            
            # Contact aliases
            "контакт": "контактна информация",
            "контакти": "контактна информация",
            "телефон за връзка": "контактна информация",
            "телефонен номер": "контактна информация",
            "адрес": "контактна информация",
            "контактна информация": "контактна информация",
            
            # Working hours aliases
            "работно време": "график на работа",
            "работите ли": "график на работа",
            "отворени ли сте": "график на работа",
            "кога работите": "график на работа",
            "до колко часа": "график на работа",
            "от колко часа": "график на работа",
            "работите ли в неделя": "график на работа",
            "работите ли в събота": "график на работа",
            "почивни дни": "график на работа",
            
            # Product aliases (телефон като продукт)
            "телефон": "каталог продукти",
            "смартфон": "каталог продукти", 
            "мобилен": "каталог продукти",
            "gsm": "каталог продукти",
            
            # Products aliases
            "продукти": "каталог продукти",
            "какви продукти": "каталог продукти",
            "каталог": "каталог продукти",
            "каталог продукти": "каталог продукти"
        }
        
        # Synonyms for query expansion (expanded for better coverage)
        self.synonyms = {
            "плаща": ["плащане", "разплащане", "пари", "цена", "такса", "карта", "превод", "начин"],
            "доставка": ["доставяне", "транспорт", "куриер", "изпращане", "пратка", "пакет", "срок"],
            "връща": ["рекламация", "възстановяване", "отмяна", "отказ", "върна", "политика"],
            "гаранция": ["сервиз", "ремонт", "поддръжка", "обслужване", "покритие", "защита"],
            "време": ["срок", "период", "дни", "часове", "минути", "колко", "кога"],
            "как": ["по какъв начин", "какво", "защо", "кога", "къде", "начин", "метод"],
            "продукт": ["стоки", "артикули", "предмети", "неща", "каталог", "асортимент"],
            "контакт": ["телефон", "адрес", "имейл", "връзка", "свързване", "информация"]
        }
        
        # Precompute embeddings for canonical queries (performance optimization)
        self.intent_embeddings = {}
        for intent, examples in self.canonical_queries.items():
            self.intent_embeddings[intent] = self.embedding_model.encode(examples)
        
        logger.success("✓ Canonical intent embeddings precomputed")
        logger.info("IntentRecognizer initialized with semantic understanding")
    
    def _auto_correct_bulgarian(self, text: str) -> str:
        """
        BUSINESS_RULE: Light auto-correction of common Bulgarian spelling mistakes
        before regex/intent analysis.
        
        Args:
            text: Query text to correct
            
        Returns:
            Auto-corrected text
        """
        corrections = {
            "редлагате": "предлагате",
            "редлагате ли": "предлагате ли",
            "вземете ли": "вземате ли",
            "асаортимент": "асортимент",
            "има ли наличен": "имате ли наличен",
            "има ли в наличност": "имате ли в наличност",
            "налични ли": "налични ли са",
            "имайте": "имате",
            "продавате ли": "продавате ли",
        }

        text_fixed = text
        for wrong, right in corrections.items():
            text_fixed = re.sub(rf"\b{wrong}\b", right, text_fixed, flags=re.IGNORECASE)

        # Light punctuation normalization
        text_fixed = re.sub(r"\s+", " ", text_fixed).strip()
        if text_fixed != text:
            logger.debug(f"Auto-corrected spelling: '{text}' → '{text_fixed}'")

        return text_fixed
    
    def _lemmatize_bulgarian(self, text: str) -> str:
        """
        BUSINESS_RULE: Lemmatize Bulgarian text using simplemma.
        
        Args:
            text: Text to lemmatize
            
        Returns:
            Lemmatized text
        """
        try:
            # Tokenize text into words first
            words = text.split()
            lemmatized_words = [simplemma.lemmatize(word, lang="bg") for word in words]
            return " ".join(lemmatized_words)
        except Exception as e:
            logger.warning(f"Failed to lemmatize Bulgarian text: {e}")
            return text
    
    def _normalize_aliases(self, query: str) -> str:
        """
        BUSINESS_RULE: Normalize query using canonical aliases with light Bulgarian stemming.
        
        Args:
            query: Original user query
            
        Returns:
            Normalized query with canonical forms
        """
        query_lower = query.lower()
        
        # Light Bulgarian stemming - remove common endings
        stemmed_query = self._light_stem_bulgarian(query_lower)
        
        # Check for exact alias matches (case-insensitive)
        original_query = query
        for alias, canonical in self.aliases.items():
            if alias in query_lower or alias in stemmed_query:
                # Replace the alias with canonical form (case-insensitive)
                import re
                query = re.sub(re.escape(alias), canonical, query, flags=re.IGNORECASE)
                logger.info(f"Normalized alias: '{alias}' -> '{canonical}' in query: '{query}'")
                break
        
        # Morphological fallback - проверка за глаголни форми
        if query == original_query:  # Ако не е намерен alias
            for base_word in ["плаща", "доставка", "връща", "гаранция", "промо", "код", "адрес"]:
                if base_word in query_lower:
                    for alias, canonical in self.aliases.items():
                        if base_word in alias:
                            import re
                            query = re.sub(re.escape(base_word), canonical, query, flags=re.IGNORECASE)
                            logger.info(f"Morphological fallback normalized '{base_word}' -> '{canonical}' in query: '{query}'")
                            break
                    break
        
        # Double-pass semantic check - предотвратява грешни нормализации
        if query != original_query:
            try:
                from sentence_transformers import util
                norm_emb = self.embedding_model.encode(query)
                orig_emb = self.embedding_model.encode(original_query)
                similarity = util.cos_sim(norm_emb, orig_emb).item()
                
                if similarity < 0.6:
                    logger.debug(f"Alias normalization changed semantics too much ({similarity:.2f}), reverting")
                    query = original_query
                else:
                    logger.debug(f"Alias normalization preserved semantics ({similarity:.2f})")
            except Exception as e:
                logger.warning(f"Semantic check failed: {e}, keeping normalized query")
        
        return query
    
    def _light_stem_bulgarian(self, text: str) -> str:
        """
        Light Bulgarian stemming - remove common endings for better matching.
        
        Args:
            text: Text to stem
            
        Returns:
            Stemmed text
        """
        # Common Bulgarian endings to remove
        endings = ['-та', '-то', '-те', '-ите', '-ата', '-ото', '-ето', '-ия', '-ията']
        
        stemmed = text
        for ending in endings:
            if stemmed.endswith(ending):
                stemmed = stemmed[:-len(ending)]
                break
        
        return stemmed
    
    def recognize_intent(self, query: str) -> Tuple[str, float, str]:
        """
        BUSINESS_RULE: Recognize user intent using semantic similarity.
        
        Args:
            query: User's original query
            
        Returns:
            Tuple of (intent_category, confidence_score, rewritten_query)
        """
        logger.debug(f"Recognizing intent for: '{query}'")
        
        # Encode the input query
        query_embedding = self.embedding_model.encode(query)
        
        best_intent = "general"
        best_score = 0.0
        best_canonical = query
        
        # Compare with precomputed canonical embeddings (optimized)
        for intent, example_embeddings in self.intent_embeddings.items():
            similarities = util.cos_sim(query_embedding, example_embeddings)
            max_score_idx = np.argmax(similarities)
            max_score = similarities[0, max_score_idx].item()
            
            # Debug: Log similarity for each intent
            logger.debug(f"Intent '{intent}': max similarity = {max_score:.3f}")
            
            if max_score > best_score:
                best_score = max_score
                best_intent = intent
                best_canonical = self.canonical_queries[intent][max_score_idx]
        
        logger.debug(f"Best intent: '{best_intent}' with score {best_score:.3f}")
        
        # Only rewrite if confidence is high enough (further lowered for Bulgarian)
        if best_score > 0.45:  # Further lowered threshold for Bulgarian models
            logger.info(f"Intent recognized: {best_intent} (confidence: {best_score:.2f})")
            return best_intent, best_score, best_canonical
        else:
            logger.debug(f"No clear intent found, using original query")
            return "general", best_score, query
    
    def expand_query(self, query: str) -> str:
        """
        BUSINESS_RULE: Expand query with semantic synonyms for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with synonyms
        """
        expanded_terms = []
        query_lower = query.lower()
        
        # Add original query
        expanded_terms.append(query)
        
        # Add synonyms for key terms with dynamic weighting
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Приоритет на ключови термини - само най-релевантните
                priority_synonyms = synonyms[:3]  # само първите 3
                for synonym in priority_synonyms:
                    if synonym not in query_lower:
                        expanded_terms.append(synonym)
        
        # Create expanded query
        if len(expanded_terms) > 1:
            expanded_query = f"{query} {' '.join(expanded_terms[1:])}"
            logger.debug(f"Query expanded: '{query}' -> '{expanded_query}'")
            return expanded_query
        
        return query
    
    def _check_product_exists_in_db(self, product_type: str) -> bool:
        """
        BUSINESS_RULE: Check if the extracted product type actually exists in database.
        
        Args:
            product_type: The product type extracted from user query
            
        Returns:
            True if product exists in database, False otherwise
        """
        try:
            from src.database.db import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if any products match the extracted product type
            sql = """
                SELECT COUNT(*) 
                FROM products 
                WHERE LOWER(name) LIKE %s 
                   OR LOWER(description) LIKE %s 
                   OR LOWER(category) LIKE %s
            """
            
            product_lower = product_type.lower()
            cursor.execute(sql, (f'%{product_lower}%', f'%{product_lower}%', f'%{product_lower}%'))
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            exists = count > 0
            logger.info(f"Product '{product_type}' exists in DB: {exists} (found {count} matches)")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking product existence: {e}")
            return False
    
    def add_context_boost(self, query: str, intent: str) -> str:
        """
        BUSINESS_RULE: Добавя контекстно подсилване към заявката.
        
        Args:
            query: Оригинална заявка
            intent: Разпознато намерение
            
        Returns:
            Заявка с контекстно подсилване
        """
        if intent in self.context_boosters:
            context_terms = self.context_boosters[intent]
            boosted_query = f"{query} {context_terms}"
            logger.debug(f"Context boosted: '{query}' -> '{boosted_query}' (intent: {intent})")
            return boosted_query
        
        return query
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        BUSINESS_RULE: Complete query processing with intent recognition and expansion.
        
        Args:
            query: User's original query
            
        Returns:
            Dictionary with processed query information
        """
        original_query = query.strip()
        query_lower = original_query.lower()
        
        # FAST-PATH FIRST: Check for products before any processing
        # This ensures we catch product queries before lemmatization changes the text
        products_match = re.search(
            r"\b(?:предлагате ли|имате ли|продавате ли|налични ли са|има ли|купя|поръчам|вземете ли)\s+([a-zA-Zа-яА-Я0-9\s-]+?)\b(?:\?|$)",
            query_lower
        )
        if products_match:
            product_type = products_match.group(1).strip()
            # Only classify as product if the word after the verb is a real product
            if any(kw in product_type.lower() for kw in [
                "лаптоп", "ноутбук", "компютър", "телефон", "смартфон", "смартфони", "gsm", 
                "часовник", "таблет", "слушалки", "мишка", "клавиатура", "монитор",
                "принтер", "скенер", "камера", "аксесоари", "периферия", "смартфон",
                "iphone", "samsung", "dell", "hp", "lenovo", "asus", "acer"
            ]):
                result = {
                    "original_query": original_query,
                    "processed_query": original_query,
                    "intent": "products",
                    "confidence": 0.95,
                    "query_type": "regex_fastpath",
                    "product_filter": product_type,
                    "expanded_terms": []
                }
                logger.info(f"✓ Regex fast-path (product='{product_type}') for '{original_query}'")
                return result
        
        # Fallback regex for general products queries
        if re.search(r"\b(какви продукти( имате)?|каталог( продукти)?|каталога|продукти|стоки)\b", query_lower):
            result = {
                "original_query": original_query,
                "processed_query": original_query,
                "intent": "products",
                "confidence": 0.95,
                "query_type": "regex_fastpath",
                "product_filter": None,
                "expanded_terms": []
            }
            logger.info(f"✓ Regex fast-path (general products) for '{original_query}'")
            return result
        
        # FAST-PATH: Working hours queries (days of week)
        if re.search(r"\b(работите ли|отворени ли сте|работно време|график|кога работите|до колко часа|от колко часа)\b", query_lower):
            # Check for specific days
            days_match = re.search(r"\b(понеделник|вторник|сряда|четвъртък|петък|събота|неделя|почивни дни|празници)\b", query_lower)
            result = {
                "original_query": original_query,
                "processed_query": original_query,
                "intent": "working_hours",
                "confidence": 0.95,
                "query_type": "regex_fastpath",
                "day_filter": days_match.group(1) if days_match else None,
                "expanded_terms": []
            }
            logger.info(f"✓ Regex fast-path (working hours) for '{original_query}'")
            return result
        
        # Step -1: Auto-correct Bulgarian spelling before regex/intent analysis
        query_lower = self._auto_correct_bulgarian(query_lower)
        
        # Step 0: Lemmatize Bulgarian words to base forms
        query_lower = self._lemmatize_bulgarian(query_lower)

        # Step 0: Normalize query with aliases
        normalized_query = self._normalize_aliases(original_query)

        # Step 1: Recognize intent (primary pass)
        intent, confidence, canonical_query = self.recognize_intent(normalized_query)

        # Step 2: Retry with original query if low confidence
        if intent == "general" and normalized_query != original_query:
            intent, confidence, canonical_query = self.recognize_intent(original_query)

        # Step 3: Fallback to keyword-based intent if still uncertain
        if confidence < 0.45:
            keyword_intents = {
                "плаща": "payment",
                "плащане": "payment", 
                "плащам": "payment",
                "разплащане": "payment",
                "доставка": "delivery",
                "доставяне": "delivery",
                "връщане": "return_policy",
                "върна": "return_policy",
                "рекламация": "return_policy",
                "гаранция": "warranty",
                "сервиз": "warranty",
                "контакт": "contact",
                "телефон": "contact",
                "продукт": "products",
                "продукти": "products",
                "стоки": "products",
                "продавате": "products",
                "асортимент": "products",
                "каталог": "products",
                "промо": "promotions"
            }
            for kw, kw_intent in keyword_intents.items():
                if kw in query_lower:
                    logger.debug(f"Keyword-based intent match: {kw} → {kw_intent}")
                    intent = kw_intent
                    confidence = 0.55
                    canonical_query = self.canonical_queries[kw_intent][0] if kw_intent in self.canonical_queries else original_query
                    break

        # Step 4: Expand query with synonyms
        expanded_query = self.expand_query(normalized_query)
        
        # Step 5: Add context boost based on intent
        if intent != "general":
            expanded_query = self.add_context_boost(expanded_query, intent)

        # Step 6: Decide final form
        if confidence > 0.7:
            final_query = canonical_query
            query_type = "canonical"
        elif len(expanded_query) > len(original_query):
            final_query = expanded_query
            query_type = "expanded"
        else:
            final_query = normalized_query
            query_type = "normalized"
        
        result = {
            "original_query": original_query,
            "processed_query": final_query,
            "intent": intent,
            "confidence": round(confidence, 3),
            "query_type": query_type,
            "expanded_terms": expanded_query.split() if expanded_query != original_query else []
        }
        
        logger.info(f"🧠 Query processed: {query_type.upper()} (intent: {intent}, conf: {confidence:.2f}) -> '{final_query}'")
        return result
    
    def get_intent_examples(self, intent: str) -> List[str]:
        """
        Get example queries for a specific intent.
        
        Args:
            intent: Intent category
            
        Returns:
            List of example queries
        """
        return self.canonical_queries.get(intent, [])
    
    def get_all_intents(self) -> List[str]:
        """
        Get all available intent categories.
        
        Returns:
            List of intent categories
        """
        return list(self.canonical_queries.keys())
