import re



def parse_answers(text, num_questions=2):
    """
    Parse the generated text to extract answers to the questions.
    Handles various formats like:
    - "Yes. No."
    - "1. Yes\n2. No"
    - "Yes\nNo"
    - etc.
    
    Args:
        text: The generated text from the model
        num_questions: Number of questions to extract answers for
    
    Returns:
        List of answers (capitalized)
    """
    text_lower = text.strip().lower()
    answers = []
    
    # Try to find numbered answers (1. ... 2. ... etc.)
    for i in range(1, num_questions + 1):
        pattern = rf'{i}[\.\)]\s*(yes|no)'
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            answers.append(match.group(1).capitalize())
    
    # If we found all numbered answers, return them
    if len(answers) == num_questions:
        return answers
    
    # Otherwise, try to find yes/no patterns in sequence
    yes_no_pattern = r'\b(yes|no)\b'
    matches = re.findall(yes_no_pattern, text_lower, re.IGNORECASE)
    
    if len(matches) >= num_questions:
        return [m.capitalize() for m in matches[:num_questions]]
    elif len(matches) > 0:
        # Fill remaining with "Unknown"
        result = [m.capitalize() for m in matches]
        result.extend(["Unknown"] * (num_questions - len(matches)))
        return result
    else:
        return ["Unknown"] * num_questions