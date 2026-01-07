#!/usr/bin/env python3
"""
simple test script, send request to sglang server
"""

import requests
import json
import sys

def test_sglang_server(host="http://localhost", port=30000, prompt=""):
    """send test request to sglang server"""
    
    url = f"{host}:{port}/generate"
    
    # build request data
    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 100,
        }
    }
    
    print(f"send request to: {url}")
    print(f"prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # extract generated text
        if "text" in result:
            print("\ngenerated text:")
            print(result["text"])
        elif "generated_text" in result:
            print("\ngenerated text:")
            print(result["generated_text"])
        
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"error: cannot connect to {url}")
        print("please ensure sglang server is running")
        return None
    except requests.exceptions.Timeout:
        print("error: request timeout")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"response content: {response.text}")
        return None
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # get prompt from command line argument
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Introduce yourself, limit your response in 50 words"
    
    # get host and port from environment variables
    import os
    host = os.getenv("SGLANG_HOST", "http://localhost")
    port = int(os.getenv("SGLANG_PORT", "30000"))
    
    test_sglang_server(host=host, port=port, prompt=prompt)

