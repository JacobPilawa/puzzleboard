import pandas as pd
import unicodedata
from rapidfuzz import fuzz, process
from helpers import load_data

'''
quick attempt at fuzzy matching unique names across events/sheets. 
massive undertaking to take this further i think, but still a decent start
'''

def normalize_name(name):
    # Lowercase, remove accents, strip whitespace
    name = name.lower().strip()
    name = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )
    return name

def find_potential_duplicates(names, threshold=85):
    normalized_names = {name: normalize_name(name) for name in names}
    duplicates = {}

    checked = set()

    for name1, norm1 in normalized_names.items():
        if name1 in checked:
            continue
        matches = []
        for name2, norm2 in normalized_names.items():
            if name1 == name2 or name2 in checked:
                continue
            score = fuzz.token_sort_ratio(norm1, norm2)
            if score >= threshold:
                matches.append((name2, score))
        if matches:
            duplicates[name1] = matches
            checked.add(name1)
            for match, _ in matches:
                checked.add(match)

    return duplicates

# Helper to get all events for a name
def get_events_for_name(df, name):
    return sorted(df[df['Name'] == name]['Event'].unique())

# Load the data
df = load_data()

# Extract unique names
names = sorted(df['Name'].unique())

# Find potential duplicates
potential_duplicates = find_potential_duplicates(names)

# Sort by the highest match score for each name
sorted_duplicates = sorted(
    potential_duplicates.items(),
    key=lambda item: max(score for _, score in item[1]),
    reverse=True
)

# Separate matches with score == 100 and others
exact_matches = []
fuzzy_matches = []

for name, matches in sorted_duplicates:
    exact = [(m, s) for m, s in matches if s == 100]
    fuzzy = [(m, s) for m, s in matches if s < 100]
    if exact:
        exact_matches.append((name, exact))
    if fuzzy:
        fuzzy_matches.append((name, fuzzy))

# Write to file
output_file = "potential_duplicates.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("==== EXACT MATCHES (score = 100) ====\n\n")
    for name, matches in exact_matches:
        f.write("=" * 60 + "\n")
        f.write(f"Name: {name}\n")
        name_events = get_events_for_name(df, name)
        f.write("  Events:\n")
        for event in name_events:
            f.write(f"    - {event}\n")
        f.write("\n")
        for match, score in matches:
            f.write(f"Original Name: {name}\n")
            f.write(f"Possible Dupe: {match} (score: {score})\n")
            match_events = get_events_for_name(df, match)
            f.write("  Events:\n")
            for event in match_events:
                f.write(f"    - {event}\n")
            f.write("\n")

    f.write("\n\n==== OTHER POTENTIAL DUPLICATES ====\n\n")
    for name, matches in fuzzy_matches:
        f.write("=" * 60 + "\n")
        f.write(f"Name: {name}\n")
        name_events = get_events_for_name(df, name)
        f.write("  Events:\n")
        for event in name_events:
            f.write(f"    - {event}\n")
        f.write("\n")
        for match, score in sorted(matches, key=lambda x: x[1], reverse=True):
            f.write(f"Original Name: {name}\n")
            f.write(f"Possible Dupe: {match} (score: {score})\n")
            match_events = get_events_for_name(df, match)
            f.write("  Events:\n")
            for event in match_events:
                f.write(f"    - {event}\n")
            f.write("\n")

print(f"Potential duplicates written to: {output_file}")