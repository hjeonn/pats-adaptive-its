import json
import logging
from models import generate_response

def classify_student_traits(conversation_history, model_dict, model_type):
    prompt = f"""Based on the following conversation, predict the student's mastery level and Big Five traits as High, Mid, or Low. Use the definitions below.

Openness
[High] Creativity in answers; open to new ideas or challenges.
[Mid] Occasionally curious or flexible.
[Low] Prefers routine responses; avoids new challenges.

Conscientiousness
[High] Careful and organized; attentive to detail.
[Mid] Moderately consistent and responsible.
[Low] Careless or inconsistent.

Extraversion
[High] Actively engages and expresses ideas openly.
[Mid] Participates when prompted.
[Low] Brief responses; limited engagement.

Agreeableness
[High] Polite, cooperative, and trustful.
[Mid] Generally cooperative.
[Low] Argumentative or skeptical.

Negative Sensitivity
[High] Appears anxious or nervous; exhibits dramatic shifts in mood.
[Mid] Moderately stable.
[Low] Emotionally stable and confident.

Return the result in strict JSON format:
{{
  "Mastery": "",
  "Big Five": {{
    "Openness": "",
    "Conscientiousness": "",
    "Extraversion": "",
    "Agreeableness": "",
    "Negative Sensitivity": ""
  }}
}}

Dialogue history:
{conversation_history}
"""
    response = generate_response(prompt, model_dict, model_type, temperature=0.1)
    response = response.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e}. Response: {response}")
        return {
            "Mastery": "Mid", 
            "Big Five": {
                "Openness": "Mid", "Conscientiousness": "Mid", 
                "Extraversion": "Mid", "Agreeableness": "Mid", "Negative Sensitivity": "Mid"
            }
        }

def determine_teaching_strategy(student_traits):
    strategies = []
    mastery = student_traits.get("Mastery", "Mid")
    big_five = student_traits.get("Big Five", {})

    def get_trait(key):
        val = big_five.get(key, "Mid")
        return "Mid" if val == "Medium" else val

    openness = get_trait("Openness")
    conscientiousness = get_trait("Conscientiousness")
    extraversion = get_trait("Extraversion")
    agreeableness = get_trait("Agreeableness")
    negative_sensitivity = get_trait("Negative Sensitivity")

    low_to_mid = ["Low", "Mid", "Medium"]
    mid_to_high = ["Mid", "Medium", "High"]

    if mastery == "Low" and negative_sensitivity in mid_to_high:
        strategies.append("Direct Repair")

    if mastery in low_to_mid and conscientiousness == "High":
        strategies.append("Display Question")
        strategies.append("Form-focused Feedback")

    if openness == "High" and mastery in mid_to_high:
        strategies.append("Referential Question")

    if mastery in ["Mid", "Medium"] and extraversion == "Low":
        strategies.append("Seeking Clarification")

    if mastery == "Low" and extraversion == "Low" and agreeableness == "High":
        strategies.append("Extended Teacher Turn")

    if mastery == "Low" and agreeableness == "High":
        strategies.append("Scaffolding: Presentation")

    if mastery in low_to_mid and conscientiousness == "Low":
        strategies.append("Scaffolding: Modeling")

    if mastery == "High" and openness == "High":
        strategies.append("Scaffolding: Extension")

    return sorted(list(set(strategies)))