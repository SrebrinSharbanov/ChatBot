#!/usr/bin/env python3
"""
Unit test for RAG chatbot with various question types.
Tests different intents and scenarios to evaluate system performance.
"""

import sys
import os
sys.path.append('/app')

import requests
import json
import time
from typing import List, Dict, Any

class ChatbotTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/query"
        
    def test_query(self, query: str) -> Dict[str, Any]:
        """Test a single query and return results"""
        payload = {
            "q": query,
            "k": 5
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "query": query,
                "answer": f"ERROR: {str(e)}",
                "score": 0,
                "sources": []
            }
    
    def run_tests(self) -> None:
        """Run comprehensive tests with various question types"""
        
        test_cases = [
            # Product queries
            {
                "category": "Продукти - общи",
                "questions": [
                    "Какви продукти предлагате?",
                    "Какво имате в каталога?",
                    "Покажете ми продуктите"
                ]
            },
            {
                "category": "Продукти - конкретни категории", 
                "questions": [
                    "Предлагате ли лаптопи?",
                    "Имате ли телефони?",
                    "Продавате ли часовници?",
                    "Имате ли таблети?"
                ]
            },
            {
                "category": "Плащане и поръчки",
                "questions": [
                    "Как мога да платя?",
                    "Може ли да платя с кредитна карта?",
                    "Предлагате ли разсрочено плащане?",
                    "Как да направя поръчка?"
                ]
            },
            {
                "category": "Доставка и проследяване",
                "questions": [
                    "Кога ще пристигне поръчката?",
                    "Как мога да проследя поръчката си?",
                    "Колко време отнема доставката?"
                ]
            },
            {
                "category": "Гаранция и сервиз",
                "questions": [
                    "Каква гаранция имате?",
                    "Къде мога да поправя продукт?",
                    "Как да направя рекламация?"
                ]
            },
            {
                "category": "Общи въпроси",
                "questions": [
                    "Как да се свържа с вас?",
                    "Къде се намирате?",
                    "Работите ли в неделя?"
                ]
            }
        ]
        
        print("=" * 80)
        print("🤖 RAG CHATBOT UNIT TESTS")
        print("=" * 80)
        print()
        
        total_tests = 0
        total_score = 0
        
        for test_group in test_cases:
            print(f"📂 {test_group['category']}")
            print("-" * 50)
            
            for question in test_group['questions']:
                total_tests += 1
                print(f"\n❓ Въпрос: {question}")
                
                result = self.test_query(question)
                score = result.get('score', 0)
                answer = result.get('answer', 'Няма отговор')
                sources = result.get('sources', [])
                
                total_score += score
                
                print(f"📊 Score: {score}/100")
                print(f"💬 Отговор: {answer}")
                
                if sources:
                    print(f"📚 Източници ({len(sources)}):")
                    for source in sources[:3]:  # Show max 3 sources
                        source_id = source.get('source_id', 'N/A')
                        relevance = source.get('relevance', 0)
                        print(f"   • {source_id} ({relevance:.1f}%)")
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.5)
            
            print("\n" + "=" * 50)
        
        # Summary
        avg_score = total_score / total_tests if total_tests > 0 else 0
        print(f"\n📈 SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   Average score: {avg_score:.1f}/100")
        print(f"   Tests above 80: {sum(1 for _ in range(total_tests) if avg_score >= 80)}")
        print("=" * 80)

def main():
    """Main function to run the tests"""
    tester = ChatbotTester()
    
    print("Starting RAG Chatbot Unit Tests...")
    print("Make sure the chatbot is running on http://localhost:8000")
    print()
    
    tester.run_tests()

if __name__ == "__main__":
    main()
