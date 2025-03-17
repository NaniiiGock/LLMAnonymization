import os
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig

# Load environment variables
load_dotenv()

# YAML configuration as a string
YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4

rails:
  config:
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - LOCATION_ADDRESS_STREET
      output:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - LOCATION_ADDRESS_STREET

  input:
    flows:
      - mask sensitive data on input

  output:
    flows:
      - mask sensitive data on output
"""

# Load the configuration
config = RailsConfig.from_content(yaml_content=YAML_CONFIG)
rails = LLMRails(config)

# Example user input
user_input = "Hello! I'm John Doe. My email is john.doe@example.com, and I live at 123 Main St. Where do I live?"

response = rails.generate(messages=[{"role": "user", "content": user_input}])

# Output the response
print("Anonymized Response:")
print(response["content"])
