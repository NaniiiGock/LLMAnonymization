# # from anonymizer import PIIAnonymizer
# # from llm_handler import LLMHandler
# # from nemoguardrails import LLMRails, RailsConfig

# # def main():
# #     print("🚀 Welcome to the PII-Protected LLM Chatbot!")

# #     config = RailsConfig.from_path("nemo/config.yml")
# #     rails = LLMRails(config)

# #     anonymizer = PIIAnonymizer()
# #     llm = LLMHandler()

# #     while True:
# #         user_input = input("\nUser: ")
# #         if user_input.lower() in ["exit", "quit"]:
# #             print("Goodbye! 👋")
# #             break

# #         anonymized_input = anonymizer.anonymize_text(user_input)
# #         print(f"\n🔒 Anonymized Input: {anonymized_input}")

# #         sanitized_input = rails.generate(messages=[{"role": "user", "content": anonymized_input}])["content"]
# #         print("SANITIZED input: ", sanitized_input)

# #         llm_response = llm.query_llm(sanitized_input)
# #         print("LLM Response: ", llm_response)

# #         sanitized_response = rails.generate(messages=[{"role": "assistant", "content": llm_response}])["content"]
# #         print("SANITIZED response: ", sanitized_response)

# #         final_response = anonymizer.de_anonymize_text(sanitized_response)
# #         print(f"\n🤖 LLM Response: {final_response}")

# # if __name__ == "__main__":
# #     main()


# from pii_detector import AdvancedPIIDetector
# from llm_handler import LLMHandler
# from nemoguardrails import LLMRails, RailsConfig

# def main():
#     print("🚀 Welcome to the AI Chatbot with Advanced PII Protection!")

#     # Load NeMo Guardrails Config
#     config = RailsConfig.from_path("nemo/config.yml")
#     rails = LLMRails(config)

#     pii_detector = AdvancedPIIDetector()
#     llm = LLMHandler()

#     while True:
#         user_input = input("\nUser: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Goodbye! 👋")
#             break

#         # Step 1: Detect PII
#         detected_pii = pii_detector.detect_pii(user_input)
#         print(f"\n🔍 Detected PII: {detected_pii}")

#         # Step 2: Anonymize user input
#         anonymized_input = pii_detector.anonymize_text(user_input)
#         print(f"\n🔒 Anonymized Input: {anonymized_input}")

#         # Step 3: Process input with NeMo Guardrails
#         sanitized_input = rails.generate(messages=[{"role": "user", "content": anonymized_input}])["content"]
#         print(f"SANITIZED input: {sanitized_input}")

#         # Step 4: Send sanitized input to LLM
#         llm_response = llm.query_llm(sanitized_input)
#         print(f"LLM Response: {llm_response}")

#         # Step 5: Process LLM response with NeMo Guardrails
#         sanitized_response = rails.generate(messages=[{"role": "assistant", "content": llm_response}])["content"]
#         print(f"SANITIZED response: {sanitized_response}")

#         # Step 6: Display the final response
#         print(f"\n🤖 LLM Response: {sanitized_response}")

# if __name__ == "__main__":
#     main()

from pii_detector import AdvancedPIIDetector
from llm_handler import LLMHandler
from nemoguardrails import LLMRails, RailsConfig

def main():
    print("🚀 Welcome to the AI Chatbot with Advanced PII Protection!")

    # Load NeMo Guardrails Config
    config = RailsConfig.from_path("nemo/config.yml")
    rails = LLMRails(config)

    pii_detector = AdvancedPIIDetector()
    llm = LLMHandler()

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        # Step 1: Detect PII
        detected_pii = pii_detector.detect_pii(user_input)
        print(f"\n🔍 Detected PII: {detected_pii}")

        # Step 2: Anonymize user input
        anonymized_input = pii_detector.anonymize_text(user_input)
        print(f"\n🔒 Anonymized Input: {anonymized_input}")

        # Step 3: Process input with NeMo Guardrails
        sanitized_input = rails.generate(messages=[{"role": "user", "content": anonymized_input}])["content"]
        print(f"SANITIZED input: {sanitized_input}")

        # Step 4: Send sanitized input to LLM
        llm_response = llm.query_llm(sanitized_input)
        print(f"LLM Response: {llm_response}")

        # Step 5: Process LLM response with NeMo Guardrails
        sanitized_response = rails.generate(messages=[{"role": "assistant", "content": llm_response}])["content"]

        # Step 6: Check for empty Guardrails response
        if not sanitized_response:
            sanitized_response = "[ERROR: Guardrails did not return a response]"

        print(f"SANITIZED response: {sanitized_response}")

        # Step 7: De-anonymize the LLM response if needed
        final_response = pii_detector.de_anonymize_text(sanitized_response)

        print(f"\n🤖 LLM Response: {final_response}")

if __name__ == "__main__":
    main()
