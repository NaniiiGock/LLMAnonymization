import spacy
from typing import Dict, Tuple, Optional
import re
from datetime import datetime
import json
from dataclasses import dataclass
import openai 
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PrivacyConfig:
    sensitivity_levels: Dict[str, int] = None
    retention_period: int = 30
    logging_enabled: bool = True
    custom_patterns: Dict[str, str] = None

class PrivacyPipeline:
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.nlp = spacy.load("en_core_web_sm")
        self.config = config or PrivacyConfig()
        self.entity_map = {}
        self.reverse_map = {}
        
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-. ]?\d{4}[-. ]?\d{4}[-. ]?\d{4}\b'
        }
        
        if self.config.custom_patterns:
            self.patterns.update(self.config.custom_patterns)
    
    def preprocess_input(self, text: str) -> Tuple[str, Dict]:

        pattern_replacements = {}
        processed_text = text
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                original = match.group(0)
                replacement = f"[{pattern_name.upper()}_{len(self.entity_map)}]"
                processed_text = processed_text.replace(original, replacement)
                self.entity_map[replacement] = original
                pattern_replacements[original] = replacement
        
        print("PATTERN REPLACEMENTS: ", pattern_replacements)

        doc = self.nlp(processed_text)
        ner_replacements = {}
        
        for ent in doc.ents:
            if ent.text not in pattern_replacements.values() and ent.text not in self.entity_map.values(): 
                replacement = f"[{ent.label_}_{len(self.entity_map)}]"
                processed_text = processed_text.replace(ent.text, replacement)
                self.entity_map[replacement] = ent.text
                ner_replacements[ent.text] = replacement

        print("NER REPLACEMENTS: ", ner_replacements)

        self.reverse_map = {v: k for k, v in self.entity_map.items()}
        self.replacements = {**pattern_replacements, **ner_replacements}
        
        return processed_text, {**pattern_replacements, **ner_replacements}
    
        
    def preprocess_task(self, task):

        for original, replacement in self.replacements.items():
            if original in task:
                processed_task = task.replace(original, replacement)
        
        return processed_task


    
    def prepare_prompt(self, anonymized_text: str, task_description: str) -> str:
        privacy_instruction = """
        Process this text while maintaining privacy. Do not attempt to:
        1. Reverse any anonymized tokens
        2. Generate or infer personal information
        3. Include specific details about anonymized entities
        """
        
        return f"{privacy_instruction}\n\nTask: {task_description}\n\nText: {anonymized_text}"
    
    async def query_llm(self, prompt: str) -> str:
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a privacy-conscious assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def postprocess_output(self, llm_output: str, context: Dict) -> str:
        processed_output = llm_output
        print("RECEIVED REPLACEMENTS: ", context)
        for original, replacement in context.items():
            if replacement in processed_output:
                processed_output = processed_output.replace(replacement, original)
        
        return processed_output

    
    
    def log_interaction(self, interaction_data: Dict):
        """
        Log interaction metadata for auditing (if enabled)
        """
        if self.config.logging_enabled:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'num_entities_detected': len(interaction_data['replacements']),
                'entity_types': list(set(r.split('_')[0][1:] for r in interaction_data['replacements'].values())),
                'processing_status': 'success'
            }

    
async def example_usage():
    config = PrivacyConfig(
        sensitivity_levels={'PERSON': 3, 'ORG': 2},
        custom_patterns={'employee_id': r'\bEMP\d{6}\b'}
    )
    pipeline = PrivacyPipeline(config)
    
    user_input = """
    My name is John Smith (employee ID: EMP123456).
    Everyone can email me at john.smith@company.com or call 123-456-7890.
    I work at Microsoft in Seattle with Sarah Johnson.

    My covorkers name is Dave Black (employee ID: EMP123445).
    Everyone can email him at dave.black@company.com or call 123-456-7890.
    He works at Microsoft in Seattle with Florin Dark.
    """
    
    try:

        anonymized_input, replacements1 = pipeline.preprocess_input(user_input)
        
        task = "Who does Dave Black work with."
        preprocessed_task = pipeline.preprocess_task(task)

        prompt = pipeline.prepare_prompt(anonymized_input, preprocessed_task)
        print("Prompt: ", prompt)

        llm_response = await pipeline.query_llm(prompt)
        print("LLM Response: ", llm_response)

        final_output = pipeline.postprocess_output(llm_response, replacements1)
        
        print("Final Output: ", final_output)

        print("Replacements: ", replacements1)
        pipeline.log_interaction({
            'replacements': replacements1,
            'input_length': len(user_input),
            'output_length': len(final_output)
        })
        
        return {
            'original_input': user_input,
            'anonymized_input': anonymized_input,
            'final_output': final_output
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    import asyncio
    results = asyncio.run(example_usage())