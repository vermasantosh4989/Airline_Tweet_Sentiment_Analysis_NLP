
# Import regular expressions
import re

# Import HTML utilities
import html


# Define text cleaning function
def clean_text(text):

    # Convert input into string
    text = str(text)

    # Convert HTML entities into normal characters
    text = html.unescape(text)

    # Convert text into lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove @mentions
    text = re.sub(r'@\w+', ' ', text)

    # Remove unwanted special characters
    text = re.sub(r"[^a-z0-9'\s]", ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Return cleaned text
    return text
