#!/usr/bin/env python3
"""
Test script to diagnose model availability and response issues.
"""

import sys
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def test_model(model_name):
    """Test a specific model with OpenRouter API."""

    api_key = os.getenv("CEO_API_KEY")
    if not api_key:
        print("Error: CEO_API_KEY not found in .env")
        return False

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lyang36/IMO25",
        "X-Title": "IMO25-Test"
    }

    # Simple test payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, world!' and nothing else."}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }

    print(f"\n🧪 Testing model: {model_name}")
    print("=" * 50)

    try:
        print("📡 Sending request to OpenRouter...")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=(10, 30))
        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ Valid JSON response received")

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"💬 Response content: '{content}'")

                    if content and content.strip():
                        print("✅ Model responded with content")
                        return True
                    else:
                        print("❌ Model returned empty content")
                        return False
                else:
                    print("❌ No choices in response")
                    print(f"Response: {json.dumps(result, indent=2)}")
                    return False

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Raw response: {response.text[:500]}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Test multiple models to find working ones."""

    models_to_test = [
        "deepseek/deepseek-r1-0528-qwen3-8b",  # Current problematic model
        "deepseek/deepseek-r1",                 # Alternative
        "deepseek/deepseek-chat",               # Another alternative
        "moonshotai/kimi-k2:free",             # Previously working model
        "anthropic/claude-3-sonnet-20240229",  # Known working model
    ]

    print("🔬 OpenRouter Model Diagnostic Tool")
    print("=" * 60)

    working_models = []
    for model in models_to_test:
        if test_model(model):
            working_models.append(model)

    print("\n📋 SUMMARY")
    print("=" * 60)
    if working_models:
        print(f"✅ Working models found: {len(working_models)}")
        for model in working_models:
            print(f"   • {model}")
    else:
        print("❌ No working models found")

    print(f"\n📝 Total models tested: {len(models_to_test)}")
    print("💡 Recommendation: Use one of the working models above")

if __name__ == "__main__":
    main()