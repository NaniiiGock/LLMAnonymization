import streamlit as st
import yaml
import os
import asyncio
import json
from privacy_pipeline import PrivacyPipeline
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Privacy Pipeline",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load default config for initialization
def load_default_config():
    default_config = {
        "processing": {
            "order": ["pattern_processor", "ner_processor", "llm_invoke", "postprocessor"]
        },
        "pattern_processor": {
            "custom_patterns": {
                "employee_id": r'\bEMP\d{6}\b',
                "account_number": r'\bACCT-\d{10}\b',
                "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                "date_of_birth": r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b'
            }
        },
        "ner_processor": {
            "model": "en_core_web_sm",
            "entity_types": ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"],
            "sensitivity_levels": {
                "PERSON": 5, "ORG": 3, "GPE": 2, "LOC": 2, "DATE": 3, "MONEY": 4
            }
        },
        "llm_invoke": {
            "provider": "openai",
            "model": "gpt-4",
            "system_prompt": "You are a assistant.",
            "temperature": 0.3,
            "max_tokens": 1000,
            "api": {
                "timeout_seconds": 30,
                "retries": 3
            }
        },
        "postprocessor": {
            "mode": "restore_original",
            "additional_filtering": False,
            "placeholder_format": "<redacted {entity_type}>"
        },
        "files": {
            "input_path": "app2/data/user_input.txt",
            "task_path": "app2/data/task_description.txt",
            "output_path": "app2/data/preprocessed_output.txt",
            "log_path": "app2/logs/pipeline.log"
        },
        "logging": {
            "enabled": True,
            "level": "INFO",
            "retention_days": 30,
            "include_timestamps": True,
            "include_entity_counts": True
        }
    }
    return default_config

# Initialize session state variables
if 'config' not in st.session_state:
    st.session_state.config = load_default_config()
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'available_blocks' not in st.session_state:
    st.session_state.available_blocks = [
        "pattern_processor",
        "ner_processor",
        "llm_invoke",
        "postprocessor"
    ]
    
if 'processing_order' not in st.session_state:
    st.session_state.processing_order = st.session_state.config["processing"]["order"]
    
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
    
if 'task_description' not in st.session_state:
    st.session_state.task_description = ""

# Save configuration to file
def save_config(config, filename="app2/running_config.yml"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return filename

# Async wrapper for running the pipeline
async def run_pipeline_async(config_path, user_input, task):
    # Create temp files for input and task
    os.makedirs("app2/data", exist_ok=True)
    
    with open(config["files"]["input_path"], 'w') as f:
        f.write(user_input)
        
    with open(config["files"]["task_path"], 'w') as f:
        f.write(task)
    
    # Initialize pipeline
    pipeline = PrivacyPipeline(config_path)
    
    # Process the pipeline
    results = await pipeline.process_pipeline(user_input, task)
    
    return results

# Streamlit run wrapper for async function
def run_pipeline(config, user_input, task):
    config_path = save_config(config)
    
    with st.spinner("Processing through privacy pipeline..."):
        results = asyncio.run(run_pipeline_async(config_path, user_input, task))
    
    return results

# Main app layout
st.title("🔒 Privacy Pipeline Chat")

# Sidebar - Pipeline Configuration
with st.sidebar:
    st.header("Pipeline Configuration")
    
    # Processing order configuration with drag and drop
    st.subheader("Processing Order")
    
    # Display current processing order
    st.write("Current Order:")
    for i, block in enumerate(st.session_state.processing_order):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if i > 0:
                if st.button("↑", key=f"up_{block}"):
                    # Move block up
                    idx = st.session_state.processing_order.index(block)
                    st.session_state.processing_order[idx], st.session_state.processing_order[idx-1] = st.session_state.processing_order[idx-1], st.session_state.processing_order[idx]
                    st.experimental_rerun()
        with col2:
            st.write(f"{i+1}. {block}")
        with col3:
            if i < len(st.session_state.processing_order) - 1:
                if st.button("↓", key=f"down_{block}"):
                    # Move block down
                    idx = st.session_state.processing_order.index(block)
                    st.session_state.processing_order[idx], st.session_state.processing_order[idx+1] = st.session_state.processing_order[idx+1], st.session_state.processing_order[idx]
                    st.experimental_rerun()
    
    # Update config
    st.session_state.config["processing"]["order"] = st.session_state.processing_order
    
    # Collapsible sections for each component configuration
    with st.expander("Pattern Processor Settings"):
        st.subheader("Custom Patterns")
        
        # Show existing patterns with option to edit or delete
        custom_patterns = st.session_state.config["pattern_processor"]["custom_patterns"]
        patterns_to_delete = []
        
        for pattern_name, pattern_regex in custom_patterns.items():
            col1, col2, col3 = st.columns([3, 6, 1])
            with col1:
                st.text(pattern_name)
            with col2:
                new_pattern = st.text_input(f"Regex for {pattern_name}", pattern_regex, key=f"pattern_{pattern_name}")
                custom_patterns[pattern_name] = new_pattern
            with col3:
                if st.button("🗑️", key=f"delete_{pattern_name}"):
                    patterns_to_delete.append(pattern_name)
        
        # Remove patterns marked for deletion
        for pattern_name in patterns_to_delete:
            del custom_patterns[pattern_name]
        
        # Add new pattern
        with st.form("add_pattern_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_pattern_name = st.text_input("New Pattern Name")
            with col2:
                new_pattern_regex = st.text_input("Regex Pattern")
            
            submitted = st.form_submit_button("Add Pattern")
            if submitted and new_pattern_name and new_pattern_regex:
                custom_patterns[new_pattern_name] = new_pattern_regex
    
    with st.expander("NER Processor Settings"):
        # Select spaCy model
        model_options = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        selected_model = st.selectbox(
            "spaCy Model",
            model_options,
            index=model_options.index(st.session_state.config["ner_processor"]["model"])
        )
        st.session_state.config["ner_processor"]["model"] = selected_model
        
        # Entity types selection
        all_entity_types = ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", 
                           "PERCENT", "CARDINAL", "ORDINAL", "QUANTITY", "PRODUCT", "EVENT", "WORK_OF_ART"]
        
        selected_entities = st.multiselect(
            "Entity Types to Process",
            all_entity_types,
            default=st.session_state.config["ner_processor"]["entity_types"]
        )
        st.session_state.config["ner_processor"]["entity_types"] = selected_entities
        
        # Sensitivity levels
        st.subheader("Sensitivity Levels (1-5)")
        sensitivity_levels = st.session_state.config["ner_processor"]["sensitivity_levels"]
        
        for entity in selected_entities:
            current_level = sensitivity_levels.get(entity, 3)
            new_level = st.slider(
                f"{entity} Sensitivity",
                min_value=1,
                max_value=5,
                value=current_level,
                key=f"sensitivity_{entity}"
            )
            sensitivity_levels[entity] = new_level
        
        # Remove entities that are no longer selected
        for entity in list(sensitivity_levels.keys()):
            if entity not in selected_entities:
                del sensitivity_levels[entity]
    
    with st.expander("LLM Settings"):
        # LLM Provider
        provider_options = ["openai", "anthropic", "local"]
        selected_provider = st.selectbox(
            "LLM Provider",
            provider_options,
            index=provider_options.index(st.session_state.config["llm_invoke"]["provider"])
        )
        st.session_state.config["llm_invoke"]["provider"] = selected_provider
        
        # Model selection based on provider
        if selected_provider == "openai":
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        elif selected_provider == "anthropic":
            model_options = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        else:
            model_options = ["local-model"]
        
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=0 if st.session_state.config["llm_invoke"]["model"] not in model_options else model_options.index(st.session_state.config["llm_invoke"]["model"])
        )
        st.session_state.config["llm_invoke"]["model"] = selected_model
        
        # System prompt
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.config["llm_invoke"]["system_prompt"]
        )
        st.session_state.config["llm_invoke"]["system_prompt"] = system_prompt
        
        # Temperature and max tokens
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.config["llm_invoke"]["temperature"]),
                step=0.1
            )
            st.session_state.config["llm_invoke"]["temperature"] = temperature
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=st.session_state.config["llm_invoke"]["max_tokens"]
            )
            st.session_state.config["llm_invoke"]["max_tokens"] = max_tokens
    
    with st.expander("Postprocessor Settings"):
        # Postprocessing mode
        mode_options = ["restore_original", "keep_anonymized", "generic_placeholders"]
        selected_mode = st.selectbox(
            "Processing Mode",
            mode_options,
            index=mode_options.index(st.session_state.config["postprocessor"]["mode"])
        )
        st.session_state.config["postprocessor"]["mode"] = selected_mode
        
        # Optional settings for placeholder format
        if selected_mode == "generic_placeholders":
            placeholder_format = st.text_input(
                "Placeholder Format",
                value=st.session_state.config["postprocessor"]["placeholder_format"]
            )
            st.session_state.config["postprocessor"]["placeholder_format"] = placeholder_format
            
            st.info("Use {entity_type} in the format string to include the entity type")
    
    # Save configuration button
    if st.button("💾 Save Configuration"):
        config_path = save_config(st.session_state.config)
        st.success(f"Configuration saved to {config_path}")

# Main chat area
col1, col2 = st.columns([2, 1])

with col2:
    # Task description panel
    st.subheader("Task Description")
    task_description = st.text_area(
        "Enter your task for the LLM:",
        value=st.session_state.task_description,
        height=100,
        key="task_input"
    )
    
    # Debug information
    with st.expander("Debug Information"):
        st.json(st.session_state.config)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

with col1:
    # Chat history display
    st.subheader("Chat")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Input for user message
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Update session state
        st.session_state.user_input = user_input
        
        # Get current task description
        task = task_description if task_description else "Process this text while maintaining privacy."
        st.session_state.task_description = task
        
        # Process through privacy pipeline
        config = st.session_state.config
        
        try:
            # Process the user input
            results = run_pipeline(config, user_input, task)
            
            # Add system response to chat
            if "final_output" in results:
                response = results["final_output"]
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Add processing details as a system message
                processing_details = f"""
                **Processing Summary:**
                - Original text processed and anonymized
                - {len(results.get('processing_steps', []))} pipeline steps completed
                - Anonymization: {len(results.get('anonymized_input', '')) > 0}
                """
                st.session_state.chat_history.append({"role": "system", "content": processing_details})
            else:
                error_msg = results.get('error', 'Unknown error occurred during processing')
                st.session_state.chat_history.append({"role": "assistant", "content": f"Processing error: {error_msg}"})
            
            # Force a rerun to update the display
            st.experimental_rerun()
            
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Privacy Pipeline App - Powered by Streamlit")