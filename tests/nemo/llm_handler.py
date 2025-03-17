from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def query_llm(self, input_text):
        
        client = OpenAI()

        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content":input_text
                }
            ]
        )
        return completion.choices[0].message.content


if __name__ == "__main__":
    handler = LLMHandler()
    input = "Hi there!"
    response = handler.query_llm(input)
    print(response)