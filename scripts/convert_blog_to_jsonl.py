#!/usr/bin/env python3
"""
Convert blog/article markdown files to JSONL conversation format for nanochat training.

Usage:
    python -m scripts.convert_blog_to_jsonl --input ./information-hub --output ./data/karaage_conversations.jsonl
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Question templates for generating user prompts
QUESTION_TEMPLATES = [
    "{title}について教えて",
    "{title}って何？",
    "{title}について説明して",
    "{title}のこと教えて",
    "{title}ってどういうこと？",
    "{title}について詳しく知りたい",
    "{title}の話をして",
    "{title}について書いて",
]

# Titles to skip (not meaningful)
SKIP_TITLE_PATTERNS = [
    r'^\d{4}[\s_-]\d{2}[\s_-]\d{2}',  # Date patterns like 2015-02-01
    r'tweets?$',  # tweets
    r'^profile',
    r'^index$',
    r'^readme$',
    r'^\d+$',  # Just numbers
]


def parse_yaml_frontmatter(content: str) -> Tuple[Dict, str]:
    """Extract YAML front matter and body from markdown content."""
    frontmatter = {}
    body = content

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            body = parts[2].strip()

            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip().strip('"').strip("'")
                    frontmatter[key] = value

    return frontmatter, body


def clean_text(text: str) -> str:
    """Aggressively clean text for training."""

    # Remove URLs completely
    text = re.sub(r'https?://[^\s\]）\)\n]+', '', text)
    text = re.sub(r'www\.[^\s\]）\)\n]+', '', text)

    # Remove hatena notation
    text = re.sub(r'\[http[^\]]*:title=([^\]]+)\]', r'\1', text)  # [url:title=text] -> text
    text = re.sub(r'\[https?://[^\]]+\]', '', text)  # [url] -> nothing
    text = re.sub(r'\[asin:[^\]]+\]', '', text)  # [asin:...] -> nothing
    text = re.sub(r'\[twitter:[^\]]+\]', '', text)
    text = re.sub(r'\[embed:[^\]]+\]', '', text)
    text = re.sub(r'\[:contents\]', '', text)
    text = re.sub(r'\[f:id:[^\]]+\]', '', text)  # hatena photo

    # Remove hatena code blocks
    text = re.sub(r'>>\n.*?\n<<', '', text, flags=re.DOTALL)
    text = re.sub(r'>\|[a-z]*\|.*?\|\|<', '', text, flags=re.DOTALL)
    text = re.sub(r'>\|[a-z]*\|', '', text)
    text = re.sub(r'\|\|<', '', text)
    text = re.sub(r'\|<', '', text)
    text = re.sub(r'>\|', '', text)

    # Remove markdown code blocks
    text = re.sub(r'```[a-z]*\n.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove image markdown
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)

    # Convert links to text only
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove hatena headings markers but keep text
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown headings markers but keep text
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove list markers
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Remove reference-style links
    text = re.sub(r'^\[[^\]]+\]:\s*.*$', '', text, flags=re.MULTILINE)

    # Clean up whitespace
    text = re.sub(r'　', ' ', text)  # full-width space to half-width
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

    # Remove lines that are just punctuation or very short
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 5 or l.strip() == '']
    text = '\n'.join(lines)

    # Remove empty brackets (leftover from link removal)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)

    # Final cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def split_into_chunks(text: str, max_length: int = 400) -> List[str]:
    """Split text into chunks of reasonable length."""
    chunks = []

    # Split by double newline or single newline
    paragraphs = re.split(r'\n+', text)

    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds max, save current and start new
        if current and len(current) + len(para) > max_length:
            if len(current) >= 50:
                chunks.append(current.strip())
            current = para
        else:
            current = current + "\n" + para if current else para

        # If current chunk is already long enough, save it
        if len(current) >= max_length:
            chunks.append(current.strip())
            current = ""

    # Don't forget the last one
    if len(current) >= 50:
        chunks.append(current.strip())

    return chunks


def should_skip_title(title: str) -> bool:
    """Check if title should be skipped."""
    title_lower = title.lower().strip()
    for pattern in SKIP_TITLE_PATTERNS:
        if re.search(pattern, title_lower):
            return True
    return False


def create_conversations(title: str, content: str) -> List[List[Dict]]:
    """Create conversations from a blog post."""
    conversations = []

    # Skip meaningless titles
    if should_skip_title(title):
        return conversations

    # Clean the content
    content = clean_text(content)

    # Skip if too short after cleaning
    if len(content) < 100:
        return conversations

    # Split into chunks
    chunks = split_into_chunks(content, max_length=400)

    if not chunks:
        return conversations

    # Create title-based questions for each chunk (more variety)
    for i, chunk in enumerate(chunks):
        if len(chunk) < 80:
            continue

        # Use title-based question with variation
        template = random.choice(QUESTION_TEMPLATES)
        question = template.format(title=title)
        conversations.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": chunk}
        ])

        # Limit conversations per article
        if len(conversations) >= 3:
            break

    return conversations


def process_markdown_file(filepath: Path) -> List[List[Dict]]:
    """Process a single markdown file and return conversations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return []

    frontmatter, body = parse_yaml_frontmatter(content)

    # Get title
    title = frontmatter.get('title', '')
    if not title:
        title = filepath.stem.replace('-', ' ').replace('_', ' ')

    # Skip if title is just a timestamp
    if not title or title.isdigit() or len(title) < 3:
        heading_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        if heading_match:
            title = heading_match.group(1)
        else:
            return []

    # Clean title too
    title = re.sub(r'https?://[^\s]+', '', title).strip()
    if len(title) < 3:
        return []

    return create_conversations(title, body)


def collect_markdown_files(input_dir: Path) -> List[Path]:
    """Recursively collect all markdown files."""
    files = list(input_dir.glob('**/*.md'))
    return files


def main():
    parser = argparse.ArgumentParser(description='Convert blog markdown to JSONL conversations')
    parser.add_argument('--input', '-i', type=str, default='./information-hub',
                        help='Input directory containing markdown files')
    parser.add_argument('--output', '-o', type=str, default='./data/karaage_conversations.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--min-length', type=int, default=50,
                        help='Minimum response length to include')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_dir} for markdown files...")
    files = collect_markdown_files(input_dir)
    print(f"Found {len(files)} markdown files")

    all_conversations = []
    processed = 0
    skipped = 0

    for filepath in files:
        convs = process_markdown_file(filepath)
        if convs:
            all_conversations.extend(convs)
            processed += 1
        else:
            skipped += 1

    # Shuffle
    random.shuffle(all_conversations)

    # Write to JSONL
    print(f"\nWriting {len(all_conversations)} conversations to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"\nConversion complete!")
    print(f"  - Files processed: {processed}")
    print(f"  - Files skipped: {skipped}")
    print(f"  - Total conversations: {len(all_conversations)}")
    print(f"\nOutput: {output_file}")


if __name__ == '__main__':
    main()
