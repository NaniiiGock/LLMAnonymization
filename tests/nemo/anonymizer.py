import re
from faker import Faker

fake = Faker()

class PIIAnonymizer:
    def __init__(self):
        self.pii_patterns = {
            "PERSON": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "PHONE_NUMBER": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
        }
        self.pii_replacements = {}

    def anonymize_text(self, text):
        """Anonymizes PII by replacing detected data with fake placeholders."""
        for entity, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                fake_value = self.generate_fake_value(entity)
                self.pii_replacements[match] = fake_value
                text = text.replace(match, fake_value)
        return text

    def de_anonymize_text(self, text):
        """Restores original PII from anonymized output (optional)."""
        for fake_value, original in self.pii_replacements.items():
            text = text.replace(original, fake_value)
        return text

    def generate_fake_value(self, entity_type):
        """Generates pseudonyms for different PII types."""
        if entity_type == "PERSON":
            return fake.name()
        elif entity_type == "EMAIL_ADDRESS":
            return fake.email()
        elif entity_type == "PHONE_NUMBER":
            return fake.phone_number()
        elif entity_type == "CREDIT_CARD":
            return "****-****-****-" + str(fake.random_int(1000, 9999))
        elif entity_type == "SSN":
            return "***-**-" + str(fake.random_int(1000, 9999))
        return "[REDACTED]"


