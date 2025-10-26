"""
Test client for Mini RAG Chatbot API.
Sends sample queries and displays results with scores and sources.
"""

import argparse
import requests
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


# Sample test queries (Bulgarian)
SAMPLE_QUERIES = [
    "Каква е политиката за връщане на продукти?",
    "Колко е гаранцията за продуктите?",
    "Как мога да проследя моята поръчка?",
    "Какви са опциите за плащане?",
    "Какви лаптопи предлагате?",
    "Колко струват безжичните слушалки?",
    "Мога ли да променя адреса за доставка?",
    "Имате ли физически магазини?",
    "Какво покрива гаранцията?",
    "Колко време отнема доставката?",
    # Edge cases (should return low score)
    "Какво е времето днес?",
    "Кой е президентът на България?",
]


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def print_result(query: str, response: Dict[str, Any], index: int = None):
    """
    Print query result in formatted way.
    
    Args:
        query: Original query
        response: API response
        index: Query index (optional)
    """
    print_separator()
    if index is not None:
        print(f"Query #{index}")
    print_separator()
    
    print(f"\n📝 Въпрос: {query}")
    print(f"\n💯 Score: {response.get('score', 0)}/100")
    print(f"🎯 Confidence: {response.get('confidence', 'unknown')}")
    
    # Answer
    answer = response.get('answer', '')
    print(f"\n✍️  Отговор:")
    print(f"   {answer}")
    
    # Sources
    sources = response.get('sources', [])
    if sources:
        print(f"\n📚 Източници ({len(sources)}):")
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5
            source_id = f"{source['source_table']}:{source['source_id']}"
            title = source.get('title', 'N/A')
            similarity = source.get('similarity', 0)
            print(f"   {i}. {source_id} - {title[:60]}... (similarity: {similarity:.3f})")
    else:
        print(f"\n📚 Източници: Няма")
    
    print()


def send_query(
    url: str,
    query: str,
    k: int = 5,
    threshold: int = None,
    timeout: int = 150
) -> Dict[str, Any]:
    """
    Send query to API.
    
    Args:
        url: API endpoint URL
        query: User query
        k: Top-k documents to retrieve
        threshold: Score threshold
        timeout: Request timeout
    
    Returns:
        API response dictionary
    """
    payload = {"q": query, "k": k}
    if threshold is not None:
        payload["threshold"] = threshold
    
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout}s")
        return {"error": "timeout"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}


def run_single_query(url: str, query: str, k: int = 5, threshold: int = None):
    """
    Run single query and print result.
    
    Args:
        url: API endpoint URL
        query: User query
        k: Top-k retrieval
        threshold: Score threshold
    """
    logger.info(f"Testing query: '{query}'")
    
    start_time = time.time()
    response = send_query(url, query, k, threshold)
    elapsed = time.time() - start_time
    
    if "error" in response:
        logger.error(f"Query failed: {response['error']}")
        return
    
    print_result(query, response)
    logger.info(f"Response time: {elapsed:.2f}s")


def run_batch_queries(
    url: str,
    queries: List[str],
    k: int = 5,
    threshold: int = None,
    delay: float = 0.5
):
    """
    Run batch of queries.
    
    Args:
        url: API endpoint URL
        queries: List of queries
        k: Top-k retrieval
        threshold: Score threshold
        delay: Delay between requests (seconds)
    """
    logger.info(f"Running {len(queries)} test queries...")
    print_separator("=")
    print(f"Mini RAG Chatbot - Test Suite")
    print_separator("=")
    print()
    
    results = []
    
    for i, query in enumerate(queries, 1):
        start_time = time.time()
        response = send_query(url, query, k, threshold)
        elapsed = time.time() - start_time
        
        if "error" not in response:
            print_result(query, response, index=i)
            results.append({
                "query": query,
                "score": response.get("score", 0),
                "confidence": response.get("confidence", "unknown"),
                "response_time": elapsed,
                "sources_count": len(response.get("sources", []))
            })
        else:
            logger.error(f"Query #{i} failed: {response['error']}")
        
        # Delay between requests
        if i < len(queries):
            time.sleep(delay)
    
    # Summary
    print_separator("=")
    print("Summary")
    print_separator("=")
    
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_time = sum(r["response_time"] for r in results) / len(results)
        high_conf = sum(1 for r in results if r["confidence"] == "high")
        low_conf = sum(1 for r in results if r["confidence"] == "low")
        
        print(f"\nTotal queries: {len(results)}")
        print(f"Average score: {avg_score:.1f}/100")
        print(f"Average response time: {avg_time:.2f}s")
        print(f"High confidence: {high_conf} ({high_conf/len(results)*100:.1f}%)")
        print(f"Low confidence: {low_conf} ({low_conf/len(results)*100:.1f}%)")
        print()
    else:
        print("\nNo successful queries")


def check_health(base_url: str):
    """
    Check API health.
    
    Args:
        base_url: Base API URL
    """
    health_url = f"{base_url}/health"
    logger.info(f"Checking health: {health_url}")
    
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        health = response.json()
        
        print("\n🏥 Health Check:")
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Retriever: {health.get('retriever', 'unknown')}")
        print(f"   Generator: {health.get('generator', 'unknown')}")
        print(f"   Version: {health.get('version', 'unknown')}")
        print()
        
        return health.get('status') == 'ok'
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Mini RAG Chatbot API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/query",
        help="API query endpoint URL"
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Single query to test (if not provided, runs batch tests)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        help="Score threshold override (default: from config)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batch queries in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip health check"
    )
    
    args = parser.parse_args()
    
    # Extract base URL for health check
    base_url = args.url.replace("/api/query", "/api")
    
    # Health check
    if not args.no_health_check:
        if not check_health(base_url):
            logger.warning("API health check failed, but continuing anyway...")
    
    # Run tests
    if args.query:
        # Single query mode
        run_single_query(args.url, args.query, args.k, args.threshold)
    else:
        # Batch mode
        run_batch_queries(args.url, SAMPLE_QUERIES, args.k, args.threshold, args.delay)


if __name__ == "__main__":
    main()

