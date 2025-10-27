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
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "query": query,
                "answer": f"ERROR: {str(e)}",
                "score": 0,
                "sources": []
            }
    
    def run_category_test(self, category_name: str) -> None:
        """Run tests for a specific category only"""
        test_cases = self._get_test_cases()
        
        # Find the specific category
        target_category = None
        for test_group in test_cases:
            if category_name.lower() in test_group['category'].lower():
                target_category = test_group
                break
        
        if not target_category:
            print(f"❌ Category '{category_name}' not found!")
            return
        
        print("=" * 80)
        print(f"🤖 RAG CHATBOT - {target_category['category'].upper()}")
        print("=" * 80)
        print()
        
        total_tests = 0
        total_score = 0
        
        print(f"📂 {target_category['category']}")
        print("-" * 50)
        
        for question in target_category['questions']:
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
        
        # Summary for category
        avg_score = total_score / total_tests if total_tests > 0 else 0
        high_score_tests = sum(1 for _ in range(total_tests) if avg_score >= 80)
        
        print(f"\n📈 CATEGORY SUMMARY:")
        print(f"   Tests: {total_tests}")
        print(f"   Average score: {avg_score:.1f}/100")
        print(f"   Success rate: {(high_score_tests/total_tests)*100:.1f}%")
        print("=" * 80)

    def _get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases"""
        return [
            # Product queries - общи
            {
                "category": "Продукти - общи",
                "questions": [
                    "Какви продукти предлагате?",
                    "Какво имате в каталога?",
                    "Покажете ми продуктите",
                    "Какво продавате?",
                    "Имате ли стоки?",
                    "Какъв е вашият асортимент?",
                    "Покажете ми наличните продукти",
                    "Какви стоки имате?"
                ]
            },
            # Продукти - конкретни категории
            {
                "category": "Продукти - конкретни категории", 
                "questions": [
                    "Предлагате ли лаптопи?",
                    "Имате ли телефони?",
                    "Продавате ли часовници?",
                    "Имате ли таблети?",
                    "Предлагате ли смартфони?",
                    "Имате ли мобилни телефони?",
                    "Продавате ли компютри?",
                    "Имате ли ноутбуци?",
                    "Предлагате ли слушалки?",
                    "Имате ли камери?",
                    "Продавате ли принтери?",
                    "Имате ли монитори?",
                    "Предлагате ли клавиатури?",
                    "Имате ли мишки?",
                    "Продавате ли дронове?"
                ]
            },
            # Плащане и поръчки
            {
                "category": "Плащане и поръчки",
                "questions": [
                    "Как мога да платя?",
                    "Може ли да платя с кредитна карта?",
                    "Предлагате ли разсрочено плащане?",
                    "Как да направя поръчка?",
                    "Какви са начините на плащане?",
                    "Приемате ли карти?",
                    "Мога ли да платя с карта?",
                    "Има ли наложен платеж?",
                    "Как се плаща?",
                    "Може ли банков превод?",
                    "Приемате ли PayPal?",
                    "Има ли разсрочено плащане?",
                    "Как да платя онлайн?",
                    "Мога ли да платя при доставка?",
                    "Какви са опциите за плащане?"
                ]
            },
            # Доставка и проследяване
            {
                "category": "Доставка и проследяване",
                "questions": [
                    "Кога ще пристигне поръчката?",
                    "Как мога да проследя поръчката си?",
                    "Колко време отнема доставката?",
                    "Кога ще дойде пратката?",
                    "Какво е времето за доставка?",
                    "Колко дни отнема доставката?",
                    "Кога ще получа поръчката?",
                    "Как да проследя пратката?",
                    "Има ли експресна доставка?",
                    "Колко време отнема до София?",
                    "Доставяте ли в чужбина?",
                    "Какви са сроковете за доставка?",
                    "Кога ще пристигне в София?",
                    "Има ли безплатна доставка?",
                    "Колко струва доставката?"
                ]
            },
            # Гаранция и сервиз
            {
                "category": "Гаранция и сервиз",
                "questions": [
                    "Каква гаранция имате?",
                    "Къде мога да поправя продукт?",
                    "Как да направя рекламация?",
                    "Има ли гаранция?",
                    "Колко е гаранцията?",
                    "Каква е гаранцията за продуктите?",
                    "Къде е сервизът?",
                    "Как да направя ремонт?",
                    "Имате ли сервиз?",
                    "Къде мога да поправя лаптопа?",
                    "Каква е гаранцията за телефона?",
                    "Има ли гаранция за слушалките?",
                    "Къде мога да поправя принтера?",
                    "Как да направя рекламация за продукт?",
                    "Имате ли сервизен център?"
                ]
            },
            # Работно време
            {
                "category": "Работно време",
                "questions": [
                    "Работите ли в неделя?",
                    "Кога работите?",
                    "Работно време",
                    "Отворени ли сте в събота?",
                    "До колко часа работите?",
                    "От колко часа отваряте?",
                    "Работите ли в почивни дни?",
                    "График на работа",
                    "Кога сте отворени?",
                    "Работите ли в празници?",
                    "Отворени ли сте днес?",
                    "Кога затваряте?",
                    "Работите ли вечер?",
                    "Имате ли почивни дни?",
                    "Кога сте затворени?"
                ]
            },
            # Контактна информация
            {
                "category": "Контактна информация",
                "questions": [
                    "Как да се свържа с вас?",
                    "Къде се намирате?",
                    "Какъв е телефонът ви?",
                    "Имате ли email?",
                    "Какъв е адресът ви?",
                    "Контактна информация",
                    "Телефон за връзка",
                    "Email адрес",
                    "Адрес на магазина",
                    "Как да ви намеря?",
                    "Къде сте?",
                    "Имате ли офис?",
                    "Как да се свържа с поддръжката?",
                    "Имате ли чат поддръжка?",
                    "Как да ви пиша?"
                ]
            },
            # Политики и условия
            {
                "category": "Политики и условия",
                "questions": [
                    "Каква е политиката за връщане?",
                    "Мога ли да върна продукт?",
                    "Как да върна стока?",
                    "Има ли връщане на парите?",
                    "Какви са условията за връщане?",
                    "Колко дни имам за връщане?",
                    "Мога ли да отменя поръчка?",
                    "Как да отменя поръчка?",
                    "Има ли такса за връщане?",
                    "Как да направя рекламация?",
                    "Какви са условията за покупка?",
                    "Има ли ограничения?",
                    "Какви са правилата?",
                    "Мога ли да променя поръчка?",
                    "Как да променя адреса?"
                ]
            },
            # Цени и промоции
            {
                "category": "Цени и промоции",
                "questions": [
                    "Имате ли промоции?",
                    "Има ли отстъпки?",
                    "Какви са цените?",
                    "Има ли намаления?",
                    "Колко струва това?",
                    "Каква е цената?",
                    "Имате ли промо код?",
                    "Има ли ваучери?",
                    "Какви са акциите?",
                    "Има ли специални оферти?",
                    "Колко струва доставката?",
                    "Има ли безплатна доставка?",
                    "Какви са таксите?",
                    "Има ли скрити такси?",
                    "Каква е общата цена?"
                ]
            },
            # Технически въпроси
            {
                "category": "Технически въпроси",
                "questions": [
                    "Как да настроя продукта?",
                    "Има ли инструкции?",
                    "Как да използвам това?",
                    "Има ли ръководство?",
                    "Как да инсталирам софтуера?",
                    "Има ли драйвери?",
                    "Как да свържа устройството?",
                    "Има ли кабели?",
                    "Как да активирам гаранцията?",
                    "Има ли софтуер?",
                    "Как да обновя системата?",
                    "Има ли фиърмуер?",
                    "Как да рестартирам устройството?",
                    "Има ли проблеми с продукта?",
                    "Как да реша проблема?"
                ]
            },
            # Edge cases - гранични случаи
            {
                "category": "Edge Cases - гранични случаи",
                "questions": [
                    "Какво е квантовото изчисление?",  # Извън компетенцията
                    "Колко е 2+2?",  # Математически въпрос
                    "Какво е времето?",  # Общ въпрос
                    "Кой е президентът?",  # Политически въпрос
                    "Как да готвя?",  # Кулинарни въпроси
                    "Какво е любовта?",  # Философски въпрос
                    "Колко е високо небето?",  # Абстрактен въпрос
                    "Как да летя?",  # Нереален въпрос
                    "Имате ли хеликоптери?",  # Несъществуващ продукт
                    "Продавате ли самолети?",  # Несъществуващ продукт
                    "Как да стана милионер?",  # Финансов съвет
                    "Какво е живота?",  # Философски въпрос
                    "Колко звезди има?",  # Астрономически въпрос
                    "Как да се науча да летя?",  # Нереален въпрос
                    "Имате ли магически пръчки?"  # Фантастичен продукт
                ]
            },
            # Сложни въпроси
            {
                "category": "Сложни въпроси",
                "questions": [
                    "Искам да купя лаптоп, но не знам кой да избера. Можете ли да ми помогнете?",
                    "Имате ли препоръки за добър телефон под 500 лева?",
                    "Кой е най-добрият продукт за студент?",
                    "Искам да направя поръчка, но имам въпроси за доставката и плащането.",
                    "Мога ли да поръчам продукт и да го върна ако не ми хареса?",
                    "Имате ли пакетни оферти за офис оборудване?",
                    "Каква е разликата между вашите продукти и конкурентите?",
                    "Имате ли съвети за избор на подходящ продукт?",
                    "Мога ли да комбинирам няколко продукта в една поръчка?",
                    "Имате ли услуга за настройка на продуктите?"
                ]
            }
        ]

    def run_tests(self) -> None:
        """Run comprehensive tests with various question types"""
        
        test_cases = self._get_test_cases()
        
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
        high_score_tests = sum(1 for _ in range(total_tests) if avg_score >= 80)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   Average score: {avg_score:.1f}/100")
        print(f"   Tests above 80: {high_score_tests}")
        print(f"   Success rate: {(high_score_tests/total_tests)*100:.1f}%")
        
        # Category breakdown
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for test_group in test_cases:
            category_tests = len(test_group['questions'])
            print(f"   {test_group['category']}: {category_tests} tests")
        
        print("=" * 80)

def main():
    """Main function to run the tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Chatbot Unit Tests')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Base URL of the chatbot API (default: http://localhost:8000)')
    parser.add_argument('--category', help='Run tests for specific category only')
    parser.add_argument('--list-categories', action='store_true', 
                       help='List all available test categories')
    
    args = parser.parse_args()
    
    tester = ChatbotTester(base_url=args.url)
    
    if args.list_categories:
        print("📋 Available test categories:")
        test_cases = tester._get_test_cases()
        for i, test_group in enumerate(test_cases, 1):
            print(f"   {i}. {test_group['category']} ({len(test_group['questions'])} tests)")
        return
    
    if args.category:
        print(f"Starting RAG Chatbot Unit Tests for category: {args.category}")
        print(f"Make sure the chatbot is running on {args.url}")
        print()
        tester.run_category_test(args.category)
    else:
        print("Starting RAG Chatbot Unit Tests...")
        print(f"Make sure the chatbot is running on {args.url}")
        print()
        tester.run_tests()

if __name__ == "__main__":
    main()
