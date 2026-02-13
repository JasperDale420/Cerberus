import os
import sys
from pathlib import Path

# Add AnyLLM to path
# dataingestion/any-llm/src
ANY_LLM_PATH = Path("../dataingestion/any-llm/src").resolve()
sys.path.append(str(ANY_LLM_PATH))

try:
    from any_llm import AnyLLM
except ImportError:
    print(f"Failed to import AnyLLM. Path: {ANY_LLM_PATH}")
    sys.exit(1)


def test_deepseek():
    # Load .env manually to avoid dependency on python-dotenv
    env_path = ANY_LLM_PATH.parent / ".env"
    if env_path.exists():
        print(f"Loading .env from {env_path}")
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val

    # Ensure API Key is present
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment.")
        return

    print("Initializing AnyLLM with 'deepseek' provider...")
    client = AnyLLM.create("deepseek")

    # User requested deepseek reasoning (deepseek-reasoner)
    model = "deepseek-reasoner"
    print(f"Sending request to {model} with temperature=1.0...")

    try:
        response = client.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Explain why the sky is blue using reasoning.",
                }
            ],
            temperature=1.0,
        )
        print("\nResponse Received:")
        if hasattr(response, "choices"):
            print(response.choices[0].message.content)
        else:
            print(str(response))
        print("\nSuccess!")

    except Exception as e:
        print(f"\nFailed: {e}")


if __name__ == "__main__":
    test_deepseek()
