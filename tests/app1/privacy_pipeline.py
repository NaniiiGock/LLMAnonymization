import spacy
from typing import Dict, Tuple, Optional
import re
from datetime import datetime
import json
from dataclasses import dataclass
import openai 
from dotenv import load_dotenv
from processing.pattern_processing import PatternProcessor
from processing.ner_processing import NERProcessor
from processing.postprocessor import PostProcessor
from providers.openai_provider import OpenAIProvider

load_dotenv()

@dataclass
class PrivacyConfig:
    sensitivity_levels: Dict[str, int] = None
    retention_period: int = 30
    logging_enabled: bool = True
    custom_patterns: Dict[str, str] = None

class PrivacyPipeline:
    def __init__(self, config: Optional[PrivacyConfig] = None):

        self.config = config or PrivacyConfig()
        self.entity_map = {}
        self.reverse_map = {}

        self.pattern_processor = PatternProcessor(self.config.custom_patterns)
        self.ner_processor = NERProcessor()
        self.post_processor = PostProcessor()
        self.openai = OpenAIProvider()
    
    def preprocess_input(self, text):

        pattern_replacements = {}
        processed_text = text
        
        processed_text, pattern_replacements, entity_map = self.pattern_processor.preprocess(processed_text)
        self.entity_map = entity_map

        processed_text, ner_replacements, entity_map = self.ner_processor.preprocess(processed_text, pattern_replacements, entity_map)

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
    
    async def invoke(self, prompt):
        return await self.openai.query_llm(prompt)
    
    def postprocess_output(self, llm_output, context):
        return self.post_processor.postprocess_output(llm_output, context)

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

    
# async def example_usage():
#     config = PrivacyConfig(
#         sensitivity_levels={'PERSON': 3, 'ORG': 2},
#         custom_patterns={'employee_id': r'\bEMP\d{6}\b'}
#     )
#     pipeline = PrivacyPipeline(config)
    
#     user_input = """
#     My name is John Smith (employee ID: EMP123456).
#     Everyone can email me at john.smith@company.com or call 123-456-7890.
#     I work at Microsoft in Seattle with Sarah Johnson.

#     My covorker's name is Dave Black (employee ID: EMP123445).
#     Everyone can email him at dave.black@company.com or call 123-456-7890.
#     He works at Microsoft in Seattle with Florin Dark.
#     """
    
#     try:

#         anonymized_input, replacements1 = pipeline.preprocess_input(user_input)
        
#         task = "Who does Dave Black work with."
#         preprocessed_task = pipeline.preprocess_task(task)

#         prompt = pipeline.prepare_prompt(anonymized_input, preprocessed_task)
#         print("Prompt: ", prompt)

#         llm_response = await pipeline.invoke(prompt)
#         print("LLM Response: ", llm_response)

#         final_output = pipeline.postprocess_output(llm_response, replacements1)
        
#         print("Final Output: ", final_output)

#         # print("Replacements: ", replacements1)
#         pipeline.log_interaction({
#             'replacements': replacements1,
#             'input_length': len(user_input),
#             'output_length': len(final_output)
#         })
        
#         return {
#             'original_input': user_input,
#             'anonymized_input': anonymized_input,
#             'final_output': final_output
#         }
        
#     except Exception as e:
#         return {'error': str(e)}

# if __name__ == "__main__":
#     import asyncio
#     results = asyncio.run(example_usage())