from google import genai
from google.genai import types

client = genai.Client(api_key="GOOGLE_GENAI_API_KEY")

def gemini_chat(model: str = "gemini-2.5-flash", prompt: str = "Explain how AI works in a few words"):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        )
    )

    return response.text