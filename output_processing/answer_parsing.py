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


def parse_ocr_answers(text, num_questions=7):
    """
    Parse the generated text to extract OCR-style answers (text/numeric values).
    Handles various formats like:
    - "1. 45.1234, -122.5678\n2. 14:23:45"
    - "1. Latitude: 45.1234, Longitude: -122.5678\n2. Time: 14:23:45"
    - etc.
    
    Args:
        text: The generated text from the model
        num_questions: Number of questions to extract answers for
    
    Returns:
        List of answers (as strings, preserving original case)
    """
    text = text.strip()
    answers = []
    
    # Try to find numbered answers (1. ... 2. ... etc.)
    for i in range(1, num_questions + 1):
        # Pattern to match numbered answers: "1. answer text" or "1) answer text"
        pattern = rf'{i}[\.\)]\s*(.+?)(?=\n\s*(?:{i+1}[\.\)]|$)|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up common prefixes like "Latitude:", "Time:", etc.
            answer = re.sub(r'^(latitude|longitude|time|altitude|speed|milliamps|current|battery|voltage)[:\s]+', '', answer, flags=re.IGNORECASE)
            answer = answer.strip()
            answers.append(answer)
    
    # If we found all numbered answers, return them
    if len(answers) == num_questions:
        return answers
    
    # If we found some but not all, fill the rest with "Unknown"
    if len(answers) > 0:
        answers.extend(["Unknown"] * (num_questions - len(answers)))
        return answers
    
    # If no numbered format, try to split by newlines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) >= num_questions:
        return lines[:num_questions]
    elif len(lines) > 0:
        result = lines[:]
        result.extend(["Unknown"] * (num_questions - len(lines)))
        return result
    else:
        return ["Unknown"] * num_questions