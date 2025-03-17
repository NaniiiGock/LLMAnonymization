import spacy
from typing import Dict, List, Tuple
import re
from datetime import datetime
import random
import hashlib

class NERAnonymizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_map = {}
        self.templates = {
            "PERSON": ["PERSON_{}", "INDIVIDUAL_{}", "USER_{}"],
            "ORG": ["ORGANIZATION_{}", "COMPANY_{}", "ENTITY_{}"],
            "GPE": ["LOCATION_{}", "PLACE_{}", "REGION_{}"],
            "DATE": ["DATE_{}", "TIME_{}", "PERIOD_{}"],
            "MONEY": ["AMOUNT_{}", "SUM_{}", "VALUE_{}"]
        }
    
    def get_consistent_replacement(self, entity_text: str, entity_type: str) -> str:
        """Generate consistent replacement for entities."""
        if entity_text not in self.entity_map:
            hash_value = int(hashlib.md5(entity_text.encode()).hexdigest(), 16)
            template = random.choice(self.templates.get(entity_type, ["ENTITY_{}"])) 
            self.entity_map[entity_text] = template.format(hash_value % 1000)
        return self.entity_map[entity_text]
    
    def context_aware_anonymize(self, text: str) -> Tuple[str, Dict]:
        """Anonymize text while preserving context and relationships."""
        doc = self.nlp(text)
        anonymized_text = text
        replacements = {}

        # Process entities from longest to shortest to avoid partial replacements
        entities = sorted(doc.ents, key=lambda x: len(x.text), reverse=True)
        
        for ent in entities:
            replacement = self.get_consistent_replacement(ent.text, ent.label_)
            anonymized_text = anonymized_text.replace(ent.text, replacement)
            replacements[ent.text] = {
                'type': ent.label_,
                'replacement': replacement,
                'context': text[max(0, ent.start_char-20):min(len(text), ent.end_char+20)]
            }
        
        return anonymized_text, replacements
    
    def semantic_anonymize(self, text: str) -> Tuple[str, List[Dict]]:
        """Anonymize text while maintaining semantic meaning."""
        doc = self.nlp(text)
        modifications = []
        anonymized_text = text

        for ent in doc.ents:
            # Analyze entity context
            surrounding_tokens = [token.text for token in doc if token.i < ent.start or token.i >= ent.end]
            
            # Create context-aware replacement
            replacement = self.get_consistent_replacement(ent.text, ent.label_)
            
            # Record modification
            modifications.append({
                'original': ent.text,
                'replacement': replacement,
                'type': ent.label_,
                'context_words': surrounding_tokens[:5]  # Store some context
            })
            
            anonymized_text = anonymized_text.replace(ent.text, replacement)
        
        return anonymized_text, modifications
    
    def intelligent_redaction(self, text: str, sensitivity_levels: Dict[str, int] = None) -> str:
        """Selectively redact text based on entity type and sensitivity."""
        if sensitivity_levels is None:
            sensitivity_levels = {
                "PERSON": 3,  # High sensitivity
                "ORG": 2,    # Medium sensitivity
                "GPE": 1,    # Low sensitivity
                "DATE": 1,
                "MONEY": 2
            }
        
        doc = self.nlp(text)
        redacted_text = text
        
        for ent in doc.ents:
            sensitivity = sensitivity_levels.get(ent.label_, 1)
            
            if sensitivity >= 3:
                # Complete redaction for high sensitivity
                replacement = "█" * len(ent.text)
            elif sensitivity == 2:
                # Partial redaction for medium sensitivity
                replacement = self.get_consistent_replacement(ent.text, ent.label_)
            else:
                # Minimal redaction for low sensitivity
                replacement = f"[{ent.label_}]"
            
            redacted_text = redacted_text.replace(ent.text, replacement)
        
        return redacted_text

def example_usage():
    anonymizer = NERAnonymizer()
    
    # Example text
    text = """John Smith works at Microsoft in Seattle. 
              He joined the company on January 15, 2020 and earns $150,000 annually.
              His colleague Sarah Johnson is based in New York."""
    
    # Demonstrate different methods
    context_aware, replacements = anonymizer.context_aware_anonymize(text)
    semantic, modifications = anonymizer.semantic_anonymize(text)
    intelligent = anonymizer.intelligent_redaction(text)
    
    return {
        "original": text,
        "context_aware": context_aware,
        "semantic": semantic,
        "intelligent": intelligent,
        "replacements": replacements
    }

if __name__ == "__main__":
    results = example_usage()
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(result)