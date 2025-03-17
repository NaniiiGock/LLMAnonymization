from typing import Dict, List, Optional, Tuple
import spacy
import json
from dataclasses import dataclass, asdict
import hashlib
import openai
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

@dataclass
class PersonalInfo:
    """Structure for storing personal information"""
    name: str
    birth_date: str
    address: str
    phone: str
    email: str
    medical_history: Optional[List[str]] = None
    employment: Optional[Dict[str, str]] = None
    education: Optional[List[Dict[str, str]]] = None
    
    def to_dict(self):
        return asdict(self)

class PersonalPrivacyManager:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.base_knowledge = None
        self.privacy_map = {}
        self.reverse_map = {}
        
        # Define sensitivity levels for different types of information
        self.sensitivity_levels = {
            'name': 3,
            'birth_date': 3,
            'address': 3,
            'phone': 3,
            'email': 3,
            'medical_history': 4,
            'employment': 2,
            'education': 2
        }
    
    def set_base_knowledge(self, personal_info: PersonalInfo):
        """Initialize the base knowledge about the person"""
        self.base_knowledge = personal_info
        self._create_privacy_mappings(personal_info.to_dict())
    
    def _create_privacy_mappings(self, info_dict: Dict):
        """Create privacy-preserving mappings for all personal information"""
        def hash_value(text: str) -> str:
            """Create a consistent hash for a value"""
            return hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Create mappings for each piece of information
        for key, value in info_dict.items():
            if isinstance(value, str):
                token = f"[{key.upper()}_{hash_value(value)}]"
                self.privacy_map[value] = token
                self.reverse_map[token] = value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        token = f"[{key.upper()}_{hash_value(item)}]"
                        self.privacy_map[item] = token
                        self.reverse_map[token] = item
                    elif isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            token = f"[{key.upper()}_{sub_key.upper()}_{hash_value(sub_value)}]"
                            self.privacy_map[sub_value] = token
                            self.reverse_map[token] = sub_value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    token = f"[{key.upper()}_{sub_key.upper()}_{hash_value(sub_value)}]"
                    self.privacy_map[sub_value] = token
                    self.reverse_map[token] = sub_value

    def process_user_query(self, query: str) -> Tuple[str, Dict]:
        """Process user query and anonymize any personal information"""
        processed_query = query
        used_replacements = {}
        
        # First replace any direct mentions of personal info
        for original, replacement in self.privacy_map.items():
            if original in query:
                processed_query = processed_query.replace(original, replacement)
                used_replacements[original] = replacement
        
        # Then use NER for any other potential personal information
        doc = self.nlp(processed_query)
        for ent in doc.ents:
            if ent.text not in used_replacements:
                token = f"[NEW_{ent.label_}_{len(self.privacy_map)}]"
                processed_query = processed_query.replace(ent.text, token)
                used_replacements[ent.text] = token
                self.privacy_map[ent.text] = token
                self.reverse_map[token] = ent.text
        
        return processed_query, used_replacements

    def prepare_context(self, query: str) -> str:
        """Prepare context for LLM with privacy-preserved base knowledge"""
        context = []
        
        # Add relevant anonymized base knowledge based on query keywords
        base_dict = self.base_knowledge.to_dict()
        for key, value in base_dict.items():
            if key in query.lower():
                if isinstance(value, str):
                    context.append(f"{key}: {self.privacy_map.get(value, value)}")
                elif isinstance(value, list):
                    context.append(f"{key}: {[self.privacy_map.get(v, v) if isinstance(v, str) else v for v in value]}")
                elif isinstance(value, dict):
                    context.append(f"{key}: {dict((k, self.privacy_map.get(v, v)) for k, v in value.items())}")
        
        return "\n".join(context)

    async def query_llm(self, processed_query: str, context: str) -> str:
        """Query LLM with privacy-preserved context and query"""
        prompt = f"""
        Context (anonymized personal information):
        {context}
        
        Query: {processed_query}
        
        Please respond to the query using only the anonymized tokens provided. 
        Do not attempt to reveal or generate actual personal information.
        """
        
        try:
            response =  openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a privacy-conscious assistant. Always use provided anonymized tokens."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

    def postprocess_response(self, response: str, used_replacements: Dict, restore_info: bool = True) -> str:
        """
        Postprocess LLM response with option to restore original information
        
        Args:
            response: The LLM response
            used_replacements: Dictionary of replacements used
            restore_info: If True, restore original information; if False, use generic descriptors
        """
        processed_response = response
        
        # Check for any accidental exposure of private information
        for original, replacement in self.privacy_map.items():
            if original in processed_response:
                processed_response = processed_response.replace(original, replacement)
        
        if restore_info:
            # Restore original information from tokens
            for token, original in self.reverse_map.items():
                if token in processed_response:
                    processed_response = processed_response.replace(token, original)
        else:
            # Convert to generic descriptors
            for token in self.reverse_map:
                if token in processed_response:
                    token_type = token.split('_')[0][1:].lower()
                    processed_response = processed_response.replace(token, f"<{token_type}>")
        
        return processed_response

async def example_usage():
    manager = PersonalPrivacyManager()
    personal_info = PersonalInfo(
        name="John Smith",
        birth_date="1990-05-15",
        address="123 Main St, Boston, MA 02108",
        phone="555-123-4567",
        email="john.smith@email.com",
        medical_history=["Type 2 Diabetes", "Hypertension"],
        employment={"company": "Tech Corp", "position": "Senior Engineer"},
        education=[
            {"degree": "BS Computer Science", "school": "MIT"},
            {"degree": "MS Data Science", "school": "Stanford"}
        ]
    )
    manager.set_base_knowledge(personal_info)

    user_query = "What's John Smith's educational background and current job at Tech Corp?"
    processed_query, used_replacements = manager.process_user_query(user_query)
    context = manager.prepare_context(processed_query)
    llm_response = await manager.query_llm(processed_query, context)
    anonymized_response = manager.postprocess_response(llm_response, used_replacements, restore_info=False)
    restored_response = manager.postprocess_response(llm_response, used_replacements, restore_info=True)
    
    return {
        'original_query': user_query,
        'processed_query': processed_query,
        'anonymized_response': anonymized_response,
        'restored_response': restored_response
    }

if __name__ == "__main__":
    import asyncio
    results = asyncio.run(example_usage())
    print(json.dumps(results, indent=2))