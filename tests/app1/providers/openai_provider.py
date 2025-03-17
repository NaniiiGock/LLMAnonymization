import openai 
from dotenv import load_dotenv

load_dotenv()

class OpenAIProvider:
    def __init__(self):
        self.model = "gpt-4"

    async def query_llm(self, prompt):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a privacy-conscious assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"