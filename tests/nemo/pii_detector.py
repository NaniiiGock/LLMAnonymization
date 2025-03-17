# import spacy
# from transformers import pipeline
# from presidio_analyzer import AnalyzerEngine
# from presidio_anonymizer import AnonymizerEngine

# class AdvancedPIIDetector:
#     def __init__(self):
#         """Initialize spaCy, Transformers, and Presidio for PII detection."""
#         self.nlp = spacy.load("en_core_web_trf")  # spaCy transformer-based NER
#         self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")  # Transformer-based NER
#         self.presidio_analyzer = AnalyzerEngine()
#         self.presidio_anonymizer = AnonymizerEngine()

#     def detect_pii_spacy(self, text):
#         """Detect PII using spaCy's transformer-based model."""
#         doc = self.nlp(text)
#         entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EMAIL"]}
#         return entities

#     def detect_pii_transformers(self, text):
#         """Detect PII using Hugging Face Transformers (BERT-based NER)."""
#         ner_results = self.ner_pipeline(text)
#         entities = {}
#         for entity in ner_results:
#             if entity["score"] > 0.85:  # High confidence threshold
#                 entities[entity["word"]] = entity["entity"]
#         return entities

#     def detect_pii_presidio(self, text):
#         """Detect PII using Microsoft Presidio (hybrid rule-based and ML-based)."""
#         results = self.presidio_analyzer.analyze(text=text, entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "CREDIT_CARD"], language="en")
#         entities = {res.text: res.entity_type for res in results}
#         return entities

#     def detect_pii(self, text):
#         """Run all PII detection methods and merge results."""
#         spaCy_pii = self.detect_pii_spacy(text)
#         transformer_pii = self.detect_pii_transformers(text)
#         # presidio_pii = self.detect_pii_presidio(text)
#         all_pii = {**spaCy_pii, **transformer_pii}  # Merge results
#         # all_pii = {**spaCy_pii, **transformer_pii, **presidio_pii}  # Merge results
#         return all_pii

#     def anonymize_text(self, text):
#         """Anonymize PII using Presidio."""
#         results = self.presidio_analyzer.analyze(text=text, entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "CREDIT_CARD"], language="en")
#         anonymized_text = self.presidio_anonymizer.anonymize(text, results)
#         return anonymized_text.text

import spacy
from transformers import pipeline

class AdvancedPIIDetector:
    def __init__(self):
        """Initialize spaCy and Hugging Face Transformers for PII detection."""
        self.nlp = spacy.load("en_core_web_trf")  # Transformer-based NER
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.pii_replacements = {}  # Store PII replacements

    def detect_pii_spacy(self, text):
        """Detect PII using spaCy's transformer-based model."""
        doc = self.nlp(text)
        return {ent.text: ent.label_ for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EMAIL"]}

    def detect_pii_transformers(self, text):
        """Detect PII using Hugging Face Transformers."""
        ner_results = self.ner_pipeline(text)
        return {entity["word"]: entity["entity"] for entity in ner_results if entity["score"] > 0.85}

    def detect_pii(self, text):
        """Run both NER models (spaCy & Transformers) and merge results."""
        spacy_pii = self.detect_pii_spacy(text)
        transformer_pii = self.detect_pii_transformers(text)
        return {**spacy_pii, **transformer_pii}  # Merge results

    def anonymize_text(self, text):
        """Replace detected PII with placeholders."""
        detected_pii = self.detect_pii(text)
        for entity, entity_type in detected_pii.items():
            placeholder = f"<{entity_type.upper()}>"
            self.pii_replacements[placeholder] = entity  # Store mapping
            text = text.replace(entity, placeholder)
        return text

    def de_anonymize_text(self, text):
        """Restore original PII from placeholders."""
        for placeholder, original in self.pii_replacements.items():
            text = text.replace(placeholder, original)
        return text

