from privacy_pipeline import PrivacyConfig, PrivacyPipeline

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

    My friend's name is Dave Black (employee ID: EMP123445).
    Everyone can email him at dave.black@company.com or call 123-456-7890.
    He works at Google in Seattle with Florin Dark.
    """
    
    anonymized_input, replacements1 = pipeline.preprocess_input(user_input)
    
    task = "Who does Dave Black work with."
    preprocessed_task = pipeline.preprocess_task(task)

    prompt = pipeline.prepare_prompt(anonymized_input, preprocessed_task)
    print("Prompt: ", prompt)

    llm_response = await pipeline.invoke(prompt)
    print("LLM Response: ", llm_response)

    final_output = pipeline.postprocess_output(llm_response, replacements1)
    
    print("Final Output: ", final_output)

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
    

if __name__ == "__main__":
    import asyncio
    results = asyncio.run(example_usage())