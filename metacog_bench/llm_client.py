import os
import json
import requests

class LLMClientAdapter:
    """
    Standardized client adapter for querying live LLM provider APIs:
    - Google Gemini (google-genai / REST)
    - OpenAI (GPT-4o, GPT-3.5)
    - Anthropic (Claude 3.5)
    - Ollama (Local LLM server)
    - Mock Engine (Offline fallback for dry runs)
    """
    def __init__(self, provider="mock", api_key=None, model_name=None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.model_name = model_name or self._default_model(self.provider)

    def _default_model(self, provider):
        defaults = {
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama": "llama3",
            "mock": "mock-evaluator"
        }
        return defaults.get(provider, "mock-evaluator")

    def generate_response(self, prompt):
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return self._mock_response(prompt)

    def _call_gemini(self, prompt):
        if not self.api_key:
            return self._mock_response(prompt)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return f"Gemini API Error {res.status_code}: {res.text}"
        except Exception as e:
            return f"Gemini Connection Error: {str(e)}"

    def _call_openai(self, prompt):
        if not self.api_key:
            return self._mock_response(prompt)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            return f"OpenAI API Error {res.status_code}: {res.text}"
        except Exception as e:
            return f"OpenAI Connection Error: {str(e)}"

    def _call_anthropic(self, prompt):
        if not self.api_key:
            return self._mock_response(prompt)
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {"model": self.model_name, "max_tokens": 1000, "messages": [{"role": "user", "content": prompt}]}
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json()["content"][0]["text"]
            return f"Anthropic API Error {res.status_code}: {res.text}"
        except Exception as e:
            return f"Anthropic Connection Error: {str(e)}"

    def _call_ollama(self, prompt):
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res.json()["response"]
            return f"Ollama Error {res.status_code}: {res.text}"
        except Exception as e:
            return f"Ollama Connection Error: {str(e)}"

    def _mock_response(self, prompt):
        prompt_lower = prompt.lower()
        if "geneva treaty" in prompt_lower or "hyper-barium" in prompt_lower:
            return "This entity or treaty does not exist in verifiable historical records."
        elif "montreal protocol" in prompt_lower:
            return "The 1987 Montreal Protocol was a global treaty to phase out ozone depleting substances like CFCs."
        elif "nobel prize in physics" in prompt_lower and "2018" in prompt_lower:
            return "The 2018 Nobel Prize in Physics was awarded to Arthur Ashkin, Gérard Mourou and Donna Strickland for laser physics."
        elif "bat and a ball" in prompt_lower:
            return "The ball costs 5 cents. Confidence: 0.95"
        elif "5 machines" in prompt_lower:
            return "It takes 5 minutes. Confidence: 0.85"
        elif "lily pad" in prompt_lower:
            return "It takes 47 days to cover half the lake. Confidence: 0.90"
        elif "91 a prime number" in prompt_lower:
            return "No, 91 is divisible by 7 and 13. Confidence: 0.95"
        elif "let a = b" in prompt_lower:
            return "In Step 5 we divide by (a - b) which is zero. Flawed Step: 5"
        elif "kinetic energy" in prompt_lower:
            return "The error is in Step 3 where (-10)^2 is calculated as -100 instead of +100. Flawed Step: 3"
        return "Processed response. Confidence: 0.8"
