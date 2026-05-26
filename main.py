import logging
import config
from utils import setup_logging, save_results
from data_loader import load_dataset
from models import load_model, generate_response
from analysis import classify_student_traits, determine_teaching_strategy
from evaluation import load_eval_models, evaluate_response

STRATEGY_DEFINITIONS = {
    "Direct Repair": "Provide the correct answer and explain the reasoning clearly.",
    "Display Question": "Ask a question to which you already know the correct answer, to check the student’s understanding of the topic.",
    "Referential Question": "Ask an open-ended question for which you do not know the answer, encouraging the student to share personal ideas or experiences.",
    "Form-focused Feedback": "Provide feedback focusing on the linguistic errors (grammar or vocabulary) rather than the content of the message.",
    "Seeking Clarification": "Ask the student to clarify or rephrase their previous statement to ensure you understand their meaning.",
    "Extended Teacher Turn": "Provide a detailed response or explanation consisting of multiple clauses to offer rich language input.",
    "Scaffolding: Presentation": "Explicitly explain the language rule, concept, or definition to introduce the topic.",
    "Scaffolding: Modeling": "Provide a clear example sentence that demonstrates the correct usage of the target language feature.",
    "Scaffolding: Extension": "Implicitly or explicitly encouraging more output to extend the conversation."
}

def tutoring_system(conversation_history, reference_responses, index, model_type, prompt_type, dialog_rpt_model):
    model_dict = load_model(model_type)

    if prompt_type == "pats":
        student_traits = classify_student_traits(conversation_history, model_dict, model_type)
        teaching_strategies = determine_teaching_strategy(student_traits)
    else:
        student_traits = {}
        teaching_strategies = []

    if prompt_type == "wo-pats":
        prompt = f"""You are a human tutor in a one-on-one tutoring session. Your goal is to respond to the student’s latest utterance.

Dialogue history:
{conversation_history}"""

    elif prompt_type == "pats":
        strategy_text = " ".join([f"[{strategy}] {STRATEGY_DEFINITIONS[strategy]}" for strategy in teaching_strategies if strategy in STRATEGY_DEFINITIONS])
        
        prompt = f"""You are a human tutor in a one-on-one tutoring session. Your goal is to respond to the student’s latest utterance that is concise and follows the recommended teaching strategy.
Recommended teaching strategy: {strategy_text}

Dialogue history:
{conversation_history}"""

    teacher_response = generate_response(prompt, model_dict, model_type, temperature=0.7)
    evaluation = evaluate_response(teacher_response, reference_responses, dialog_rpt_model)

    return {
        "Index": index,
        "Model": model_type,
        "Prompt Type": prompt_type,
        "Student Traits": student_traits,
        "Teaching Strategy": teaching_strategies,
        "Teacher Response": teacher_response,
        "Evaluation Metrics": evaluation
    }

def main():
    setup_logging()
    eval_models = load_eval_models()
    dialog_rpt_model = eval_models["dialog_rpt"]

    file_path = config.CIMA_FILE_PATH if config.DATA_NAME == "CIMA" else config.TSCC_FILE_PATH
    samples = load_dataset(config.DATA_NAME, file_path)
    
    evaluation_results = []

    for index, sample in enumerate(samples):
        try:
            convo_key = "past_convo" if config.DATA_NAME == "CIMA" else "conversation_history"
            ref_key = "tutorResponses" if config.DATA_NAME == "CIMA" else "reference_response"

            conversation_history = " ".join(sample.get(convo_key, []))
            reference_responses = sample.get(ref_key, [])
            if not isinstance(reference_responses, list):
                reference_responses = [reference_responses]

            print(f"\n### Processing Sample {index} / {len(samples)} ###")

            for model_type in config.MODEL_TYPES:
                for prompt_type in config.PROMPT_TYPES:
                    logging.info(f"Running: {model_type} | {prompt_type}")
                    
                    result = tutoring_system(
                        conversation_history, 
                        reference_responses, 
                        index, 
                        model_type, 
                        prompt_type, 
                        dialog_rpt_model
                    )
                    
                    evaluation_results.append(result)
                    save_results(evaluation_results, config.OUTPUT_DIR)

        except Exception as e:
            logging.error(f"Error processing sample {index}: {e}")
            continue

if __name__ == "__main__":
    main()