import json

from tqdm import tqdm

total = 0
def remove_conversation_duplicates(conversations: list[dict]) -> list[dict]:
    if len(conversations) < 3:
        return conversations

    filtered = []
    i = 0
    while i < len(conversations):
        current = conversations[i]
        removal_type = None
        skip_count = 1

        # Scenario 1: Same content repeating
        if i + 2 < len(conversations):
            messages = conversations[i:i+3]
            if all(
                messages[0]['role'] == msg['role'] and 
                _content_similarity(messages[0]['content'], msg['content']) > 0.9
                for msg in messages[1:3]
            ):
                removal_type = "repeated_content"
                skip_count = 3

        # Scenario 2: Error pattern
        if not removal_type and i + 2 < len(conversations):
            messages = conversations[i:i+3]
            if all(msg['role'] == 'assistant' for msg in messages):
                if all(
                    'error' in msg['content'].lower() or 
                    'exception' in msg['content'].lower() or
                    'syntax error' in msg['content'].lower()
                    for msg in messages
                ):
                    removal_type = "error_pattern"
                    skip_count = 3

        # Scenario 3: Monologue
        if not removal_type and i + 2 < len(conversations):
            messages = conversations[i:i+3]
            if (all(msg['role'] == messages[0]['role'] for msg in messages) and
                all(_content_similarity(messages[0]['content'], msg['content']) > 0.7
                    for msg in messages[1:3])):
                removal_type = "monologue"
                skip_count = 3

        # Scenario 4: Alternating pattern
        if not removal_type and i + 5 < len(conversations):
            messages = conversations[i:i+6]
            if (messages[0]['role'] == messages[2]['role'] == messages[4]['role'] and
                messages[1]['role'] == messages[3]['role'] == messages[5]['role'] and
                messages[0]['role'] != messages[1]['role'] and
                _content_similarity(messages[0]['content'], messages[2]['content']) > 0.8 and
                _content_similarity(messages[2]['content'], messages[4]['content']) > 0.8 and
                _content_similarity(messages[1]['content'], messages[3]['content']) > 0.8 and
                _content_similarity(messages[3]['content'], messages[5]['content']) > 0.8):
                removal_type = "alternating_pattern"
                skip_count = 6
                
        filtered.append(current)
        
        if removal_type:
            i += skip_count
        else:
            i += 1

    global total
    if len(conversations) - len(filtered):
        total += 1
        print(total)
    return filtered

def _content_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
        
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

if __name__ == "__main__":
    with open("trajectories.json", "r") as f:
        dataset = json.load(f)
    results = []
    for data in tqdm(dataset):
        results.append({"input": remove_conversation_duplicates(data["input"])})
    with open("loop_trajs.json", "w") as f:
        json.dump(results, f, indent=2)