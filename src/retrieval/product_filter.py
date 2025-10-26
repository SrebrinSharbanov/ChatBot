"""
Product filtering and keyword extraction for intelligent product retrieval.
Analyzes user queries to extract product categories and keywords.
"""

from typing import List, Dict, Any, Tuple
from loguru import logger


class ProductFilter:
    """
    BUSINESS_RULE: Intelligent product filtering based on query keywords.
    
    Implements hybrid retrieval with category expansion:
    1. First tries keyword-based SQL filtering (with synonyms + category expansion)
    2. Falls back to vector/embedding search if no results
    3. Combines both approaches for better accuracy
    """
    
    # Дефиниране на известни продуктови категории и синоними
    PRODUCT_KEYWORDS = {
        "продукти": ["продукти", "каталог", "стоки", "асаортимент", "предлагате", "имате", "продавате", "какви", "какво", "каква", "какви стоки", "какво предлагате", "какви продукти", "какви продукти предлагате"],
        "телефон": ["телефон", "смартфон", "мобилен", "gsm", "смарт телефон", "айфон", "iphone", "samsung", "xiaomi", "nokia", "pixel"],
        "лаптоп": ["лаптоп", "ноутбук", "ultrabook", "notebook", "преносим компютър", "преносим", "преносими", "преносими компютри", "ултрабук", "dell", "lenovo", "asus", "hp", "macbook"],
        "компютър": ["компютър", "пк", "desktop", "стационарен", "tower", "компютри", "компютърни", "desktops"],
        "часовник": ["часовник", "смарт часовник", "smartwatch", "wearable", "apple watch"],
        "таблет": ["таблет", "ipad", "планшет", "планшетка", "galaxy tab"],
        "обувки": ["обувки", "кросовки", "маратонки", "маратонка", "пантофи", "ботуши", "adidas", "nike"],
        "дрехи": ["дрехи", "облекло", "тениска", "панталон", "раирана", "яке", "пуловер", "риза"],
        "градински": ["градински", "градина", "цветя", "растения", "насадки", "инструменти"],
        "слушалки": ["слушалки", "наушници", "headphones", "earbuds", "sony", "bose"],
        "камера": ["камера", "фотоапарат", "видеокамера", "дрон", "canon", "nikon"],
    }
    
    # BUSINESS_RULE: Category expansion mapping за разширено съвпадение
    # Помага при английски или различни назови на категории в базата
    CATEGORY_MAP = {
        "продукти": ["всички", "all", "каталог", "catalog", "стоки", "products", "items", "goods", "merchandise"],
        "телефон": ["телефони", "смартфони", "мобилни", "mobile", "phones", "smartphones", "cellphone", "electronics", "devices"],
        "лаптоп": ["лаптопи", "ноутбуци", "notebooks", "computers", "portable", "computing", "laptop", "ultrabook"],
        "компютър": ["компютри", "пк", "desktops", "workstations", "computing", "computers", "desktop computers", "стационарни"],
        "часовник": ["часовници", "wearables", "аксесоари", "accessories", "wearable"],
        "таблет": ["таблети", "планшети", "tablets", "ipad", "mobile"],
        "обувки": ["footwear", "shoes", "sports", "sportswear"],
        "дрехи": ["облекло", "apparel", "clothing", "fashion", "wear"],
        "градински": ["градина", "garden", "outdoor", "tools", "equipment"],
        "слушалки": ["audio", "sound", "headphone", "accessory"],
        "камера": ["камери", "photography", "imaging", "electronics"],
    }
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """
        BUSINESS_RULE: Extract product keywords with normalization and regex.
        Handles plural forms: лаптопи→лаптоп, телефони→телефон.
        Also catches "предлагате ли ..." phrases.
        """
        import re
        q = query.lower().strip()

        # Normalization: plural → singular
        norms = {
            r"\bлаптопи\b": "лаптоп",
            r"\bтелефони\b": "телефон",
            r"\bкомпютри\b": "компютър",
            r"\bчасовници\b": "часовник",
            r"\bтаблети\b": "таблет",
        }
        for pat, repl in norms.items():
            q = re.sub(pat, repl, q)

        matched = set()

        # Explicit product phrases (including "предлагате ли ...")
        explicit = [
            r"\bпредлагате ли\s+(лаптоп|компютър|телефон|часовник|таблет|слушалки)\b",
            r"\bимате ли\s+(лаптоп|компютър|телефон|часовник|таблет|слушалки)\b",
            r"\bпродавате ли\s+(лаптоп|компютър|телефон|часовник|таблет|слушалки)\b",
            r"\bкакви продукти\b", r"\bкакво предлагате\b", r"\bкаталог\b",
        ]
        for pat in explicit:
            if re.search(pat, q):
                # Map to canonical categories
                m = re.search(r"(лаптоп|компютър|телефон|часовник|таблет|слушалки)", q)
                if m:
                    matched.add(m.group(1))
                else:
                    matched.add("продукти")

        # Full dictionary with synonyms
        for cat, syns in ProductFilter.PRODUCT_KEYWORDS.items():
            for s in syns:
                if re.search(rf"\b{s}\b", q):
                    matched.add(cat)
                    break

        result = list(matched)
        logger.debug(f"Extracted keywords from query: {result}")
        return result
    
    @staticmethod
    def build_filter_query(keywords: List[str]) -> str:
        """
        BUSINESS_RULE: Build WHERE condition with strict priority:
        1. Category match (highest priority)
        2. Name/Description match
        
        Ensures we catch the right products for specific queries.
        """
        if not keywords:
            return ""
        
        # Category conditions (highest priority)
        category_conditions = []
        text_conditions = []
        
        for keyword in keywords:
            synonyms = ProductFilter.PRODUCT_KEYWORDS.get(keyword, []) + [keyword]
            category_expansions = ProductFilter.CATEGORY_MAP.get(keyword, [])
            all_words = list(dict.fromkeys(synonyms + category_expansions))
            
            # Priority 1: Category match
            for word in all_words:
                category_conditions.append(f"LOWER(category) LIKE LOWER('%{word}%')")
            
            # Priority 2: Name/Description match
            for word in all_words:
                text_conditions.append(f"(LOWER(name) LIKE LOWER('%{word}%') OR LOWER(description) LIKE LOWER('%{word}%'))")
        
        # Deduplicate
        category_conditions = list(dict.fromkeys(category_conditions))
        text_conditions = list(dict.fromkeys(text_conditions))
        
        # BUSINESS_RULE: For specific categories like "лаптоп", prioritize category match
        # Only include name/description match if no category match is found
        if any(kw in ["лаптоп", "телефон", "часовник", "таблет"] for kw in keywords):
            # For specific categories - prioritize category match
            where_clause = " OR ".join(category_conditions) if category_conditions else " OR ".join(text_conditions)
        else:
            # For general categories - combine both
            all_conditions = category_conditions + text_conditions
            where_clause = " OR ".join(all_conditions)
        
        logger.debug(f"Built filter with {len(category_conditions)} category + {len(text_conditions)} text conditions")
        return f"WHERE ({where_clause})"
    
    @staticmethod
    def format_products_for_llm(products: List[Tuple[str, str, str, float]], include_category: bool = True) -> str:
        """
        BUSINESS_RULE: Format database products for LLM processing.
        
        Converts raw product tuples into a structured text format that LLM can
        analyze and transform into natural language responses.
        
        Args:
            products: List of product tuples (name, description, category, price)
            include_category: Whether to include category in output
        
        Returns:
            Formatted product list for LLM prompt
        """
        if not products:
            return "Няма налични продукти от този тип."
        
        # Limit to first 5 products to avoid prompt being too long
        limited_products = products[:5]
        
        product_lines = []
        for i, (name, description, category, price) in enumerate(limited_products, 1):
            # Кратко описание (първите 100 символа)
            short_desc = description[:100] + "..." if len(description) > 100 else description
            
            if include_category:
                product_lines.append(
                    f"• {name} ({category}) - {price:.2f} лв\n  {short_desc}"
                )
            else:
                product_lines.append(
                    f"• {name} - {price:.2f} лв\n  {short_desc}"
                )
        
        formatted = "\n".join(product_lines)
        
        # Add note if more products exist
        if len(products) > 5:
            formatted += f"\n\n... и още {len(products) - 5} продукта в каталога."
        
        logger.debug(f"Formatted {len(limited_products)} products for LLM (total: {len(products)})")
        
        return formatted
    
    @staticmethod
    def create_llm_prompt(query: str, products_text: str, is_filtered: bool = True) -> str:
        """
        BUSINESS_RULE: Create a prompt for LLM to generate natural response.
        
        Constructs a structured prompt that instructs the LLM to transform
        product data into a conversational, natural language response.
        The prompt emphasizes natural Bulgarian language and helpful context.
        
        Args:
            query: Original user query
            products_text: Formatted product list
            is_filtered: Whether products are filtered or general catalog
        
        Returns:
            Prompt string for LLM
        """
        if is_filtered:
            prompt = f"""Потребителят пита: "{query}"

В нашия каталог имаме следните продукти:
{products_text}

Отговори на български по приятелски начин. Започни с "Да, имаме..." и опиши наличните продукти. Дай 2-3 примера с цена и кратко описание. Използвай кратки, ясни изречения. Звучи естествено като консултант.

Отговор:"""
        else:
            prompt = f"""Потребителят пита: "{query}"

В нашия каталог имаме разнообразен избор от следните категории:
{products_text}

Отговори на български по приятелски начин. Започни с "Да, имаме разнообразен каталог...". Обобщи основните категории и спомени 2-3 примера с цена. Използвай кратки, ясни изречения. Звучи естествено като консултант.

Отговор:"""
        return prompt
    
    @staticmethod
    def should_use_fallback(keyword_results: List[Tuple[str, str, str, float]], 
                       min_results: int = 2, keywords: List[str] = None) -> bool:
        """
        BUSINESS_RULE: Determine if keyword search is sufficient.

        For important categories like computers, laptops, phones,
        we never fall back to vector search if at least one keyword result exists.
        """
        if keywords and any(k in ["телефон", "лаптоп", "компютър", "часовник"] for k in keywords):
            if len(keyword_results) > 0:
                logger.debug(f"Keyword search sufficient for important category {keywords} - skipping fallback")
                return False
            else:
                min_results = 3  # fallback only if no results at all
        
        # For other categories, fallback if too few results
        if len(keyword_results) >= min_results:
            logger.debug(f"Keyword search found {len(keyword_results)} results - OK")
            return False
        
        logger.debug(f"Keyword search found only {len(keyword_results)} results - using fallback")
        return True
    
    @staticmethod
    def merge_results(keyword_results: List[Tuple[str, str, str, float]], 
                     vector_results: List[Dict[str, Any]]) -> List[Tuple[str, str, str, float]]:
        """
        BUSINESS_RULE: Merge and rank keyword and vector search results.

        Filters vector results strictly by extracted query keywords to avoid irrelevant items.
        """
        if not vector_results:
            return keyword_results
        
        merged = {}
        
        # Add keyword results first (highest priority)
        for name, desc, cat, price in keyword_results:
            merged[name.lower()] = {
                "name": name,
                "description": desc,
                "category": cat,
                "price": price,
                "score": 1.0,
                "source": "keyword"
            }
        
        # Extract keywords from the query for strict filtering
        query_keywords = []
        if keyword_results:
            # Build a combined string of all keyword product names to extract keywords
            combined_names = " ".join([name for name, _, _, _ in keyword_results])
            query_keywords = [kw.lower() for kw in ProductFilter.extract_keywords(combined_names)]
        
        filtered_vector_results = []
        if query_keywords:
            for result in vector_results:
                title = result['title'].lower()
                text = result.get('text', '').lower()
                category = result.get('source_table', '').lower()
                
                # Include only vector results that match extracted query keywords
                if any(kw in title or kw in text or kw in category for kw in query_keywords):
                    filtered_vector_results.append(result)
        else:
            filtered_vector_results = vector_results
        
        # Add vector results with reduced priority
        for result in filtered_vector_results:
            key = result['title'].lower()
            if key not in merged:
                merged[key] = {
                    "name": result['title'],
                    "description": result.get('text', '')[:200],
                    "category": result.get('source_table', 'products'),
                    "price": 0.0,
                    "score": result.get('similarity', 0.5) * 0.75,
                    "source": "vector"
                }
        
        # Sort by score descending
        sorted_results = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
        
        return [(item['name'], item['description'], item['category'], item['price']) 
                for item in sorted_results]
