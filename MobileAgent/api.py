import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _normalize_api_url(api_url: str) -> str:
    """Normalize API URL to ensure it ends with /v1/chat/completions or similar endpoint."""
    endpoint = str(api_url or "").strip()
    if not endpoint:
        raise ValueError("API URL must be provided.")
    endpoint = endpoint.rstrip("/")
    lowered = endpoint.lower()
    suffixes = ("/chat/completions", "/completions", "/responses")
    if any(lowered.endswith(sfx) for sfx in suffixes):
        return endpoint
    if lowered.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def inference_chat(chat, model, api_url, token):    
    # Normalize API URL to ensure it has the correct endpoint
    normalized_url = _normalize_api_url(api_url)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234,
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    while True:
        try:
            res = requests.post(normalized_url, headers=headers, json=data, timeout=120)
            res.raise_for_status()  # Raise an exception for bad status codes
            res_json = res.json()
            res_content = res_json['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.text
                    print(f"Response: {error_detail}")
                    # Try to parse JSON error if available
                    try:
                        error_json = e.response.json()
                        if 'error' in error_json:
                            print(f"Error details: {error_json['error']}")
                    except:
                        pass
            except Exception as ex:
                print(f"Failed to parse error response: {ex}")
            print(f"Request URL: {normalized_url}")
            # Continue loop to retry (original behavior)
        except (KeyError, IndexError) as e:
            print(f"Unexpected response format: {e}")
            try:
                print(f"Response JSON: {res_json}")
            except:
                pass
            # Continue loop to retry (original behavior)
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Continue loop to retry (original behavior)
        else:
            break
    
    return res_content
def inference_reasoning_chat(chat, model, api_url, token):
    # Normalize API URL to ensure it has the correct endpoint
    normalized_url = _normalize_api_url(api_url)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234,
        "thinking":{
            "type": "enabled",
            "budget_tokens": 2048,
        }
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    while True:
        try:
            res = requests.post(normalized_url, headers=headers, json=data, timeout=120)
            res.raise_for_status()  # Raise an exception for bad status codes
            res_json = res.json()
            res_content = res_json['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.text
                    print(f"Response: {error_detail}")
                    # Try to parse JSON error if available
                    try:
                        error_json = e.response.json()
                        if 'error' in error_json:
                            print(f"Error details: {error_json['error']}")
                    except:
                        pass
            except Exception as ex:
                print(f"Failed to parse error response: {ex}")
            print(f"Request URL: {normalized_url}")
            # Continue loop to retry (original behavior)
        except (KeyError, IndexError) as e:
            print(f"Unexpected response format: {e}")
            try:
                print(f"Response JSON: {res_json}")
            except:
                pass
            # Continue loop to retry (original behavior)
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Continue loop to retry (original behavior)
        else:
            break
    
    return res_content