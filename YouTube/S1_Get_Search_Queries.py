import requests

def generate_text(prompt: str, model: str = "tinyllama"):
    """
    Sends a prompt to the mlvoca free LLM API and prints the generated text.

    Args:
        prompt (str): Your input question or text.
        model (str): The model to use ("tinyllama" or "deepseek-r1:1.5b").
    """
    url = "https://mlvoca.com/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False  # non-streaming response
    }

    response = requests.post(url, json=body)
    data = response.json()

    # The API returns JSON with a "response" field
    # showing the model’s answer.
    text = data.get("response", "")
    print("Generated text:", text)

# Example usage
if __name__ == "__main__":
    generate_text("Explain recursion in simple terms.")
