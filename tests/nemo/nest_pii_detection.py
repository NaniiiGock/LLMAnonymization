import nest_asyncio
nest_asyncio.apply()

import os
from dotenv import load_dotenv
load_dotenv()
from nemoguardrails import LLMRails, RailsConfig


os.environ["PAI_API_KEY"] = "e66fcf3df2ec423bb54a8ed384d41a30"  # Visit https://portal.private-ai.com to get your API key

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4o

rails:
  config:
    privateai:
      server_endpoint: https://api.private-ai.com/community/v4/process/text 
      input:
        entities:
          - NAME_FAMILY
          - LOCATION_ADDRESS_STREET
          - EMAIL_ADDRESS
      output:
        entities:
          - NAME_FAMILY
          - LOCATION_ADDRESS_STREET
          - EMAIL_ADDRESS
  input:
    flows:
      - detect pii on input
      - allow llm passthrough

  output:
    flows:
      - detect pii on output
      - allow llm passthrough  
"""



config = RailsConfig.from_content(yaml_content=YAML_CONFIG)
rails = LLMRails(config)


response = rails.generate(messages=[{"role": "user", "content": "Hello! I'm John. My email id is text@gmail.com. I live in California, USA."}])

info = rails.explain()

print("Response")
print("----------------------------------------")
print(response["content"])


print("\n\nColang history")
print("----------------------------------------")
print(info.colang_history)

print("\n\nLLM calls summary")
print("----------------------------------------")
info.print_llm_calls_summary()


# response = rails.generate(messages=[{"role": "user", "content": "give me a sample email id"}])

# info = rails.explain()

# print("Response")
# print("----------------------------------------\n\n")
# print(response["content"])


# print("\n\nColang history")
# print("----------------------------------------")
# print(info.colang_history)

# print("\n\nLLM calls summary")
# print("----------------------------------------")
# info.print_llm_calls_summary()


# print("\n\nCompletions where PII was detected!")
# print("----------------------------------------")
# print(info.llm_calls[0].completion)




# # PII MASKING

# os.environ["PAI_API_KEY"] = "e66fcf3df2ec423bb54a8ed384d41a30"  # Visit https://portal.private-ai.com to get your API key

# YAML_CONFIG = """
# <!-- models:
#   - type: main
#     engine: openai
#     model: gpt-3.5-turbo-instruct -->

# models:
#   - type: main
#     engine: ollama
#     model: llama3.2
#     parameters:
#       base_url: http://localhost:11434

# rails:
#   config:
#     privateai:
#       server_endpoint: https://api.private-ai.com/cloud/v3/process/text
#       input:
#         entities:
#           - LOCATION
#           - EMAIL_ADDRESS
#   input:
#     flows:
#       - mask pii on input
# """



# config = RailsConfig.from_content(yaml_content=YAML_CONFIG)
# rails = LLMRails(config)

# response = rails.generate(messages=[{"role": "user", "content": "Hello! I'm John. My email id is text@gmail.com. I live in California, USA."}])

# info = rails.explain()

# print("Response")
# print("----------------------------------------")
# print(response["content"])


# print("\n\nColang history")
# print("----------------------------------------")
# print(info.colang_history)

# print("\n\nLLM calls summary")
# print("----------------------------------------")
# info.print_llm_calls_summary()

# os.environ["PAI_API_KEY"] = "YOUR PRIVATE AI API KEY"  # Visit https://portal.private-ai.com to get your API key

# YAML_CONFIG = """
# <!-- models:
#   - type: main
#     engine: openai
#     model: gpt-3.5-turbo-instruct -->

# models:
#   - type: main
#     engine: ollama
#     model: llama3.2
#     parameters:
#       base_url: http://localhost:11434

# rails:
#   config:
#     privateai:
#       server_endpoint: https://api.private-ai.com/cloud/v3/process/text
#       output:
#         entities:
#           - LOCATION
#           - EMAIL_ADDRESS
#   output:
#     flows:
#       - mask pii on output
# """



# config = RailsConfig.from_content(yaml_content=YAML_CONFIG)
# rails = LLMRails(config)

# response = rails.generate(messages=[{"role": "user", "content": "give me a sample email id"}])

# info = rails.explain()

# print("Response")
# print("----------------------------------------\n\n")
# print(response["content"])


# print("\n\nColang history")
# print("----------------------------------------")
# print(info.colang_history)

# print("\n\nLLM calls summary")
# print("----------------------------------------")
# info.print_llm_calls_summary()


# print("\n\nCompletions where PII was detected!")
# print("----------------------------------------")
# print(info.llm_calls[0].completion)

