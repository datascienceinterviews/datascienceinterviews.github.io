import re
import json
import os
import logging
from pathlib import Path

log = logging.getLogger("mkdocs")

def on_post_build(config, **kwargs):
    """
    Extracts interview questions from markdown files and generates a JSON file.
    """
    site_dir = config['site_dir']
    docs_dir = config['docs_dir']
    
    questions = []
    
    # Define the directory to search for interview questions
    # Based on file structure: docs/Interview-Questions/*.md
    questions_dir = Path(docs_dir) / "Interview-Questions"
    
    if not questions_dir.exists():
        log.warning(f"Questions directory not found: {questions_dir}")
        return

    # Regex patterns
    # Matches '### Question text'
    # Captures the text after ###
    question_pattern = re.compile(r"^###\s+(.+)$")
    
    # Matches metadata line: **Difficulty:** ...
    metadata_pattern = re.compile(r"\*\*Difficulty:\*\*\s*(.+?)\s*\|\s*\*\*Tags:\*\*\s*(`.+?`)\s*\|\s*\*\*Asked by:\*\*\s*(.+)")
    
    # Matches answer block start (View Answer, Answer, or just success block)
    answer_start_pattern = re.compile(r"^\?\?\? success")

    # Prepare Markdown converter
    # We try to use the same extensions as the MkDocs config
    try:
        import markdown
        md_extensions = config.get('markdown_extensions', [])
        # MkDocs adds 'tables' and 'fenced_code' by default, but they might not be in the config list
        if 'tables' not in md_extensions:
            md_extensions.append('tables')
        if 'fenced_code' not in md_extensions:
            md_extensions.append('fenced_code')
            
        md_extension_configs = config.get('mdx_configs', {})
        md = markdown.Markdown(extensions=md_extensions, extension_configs=md_extension_configs)
    except ImportError as e:
        log.warning(f"Could not import markdown module: {e}. Answers will be raw text.")
        md = None
    except Exception as e:
        log.error(f"Unexpected error initializing markdown: {e}")
        md = None

    for file_path in questions_dir.glob("*.md"):
        if file_path.name == "Interview-Questions.md": # Skip index or unwanted files
            continue
            
        topic = file_path.stem.replace("-", " ")
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_question = None
        capturing_answer = False
        answer_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # 1. Detect Question Header
            match_q = question_pattern.match(line)
            if match_q:
                # Save previous question if exists
                if current_question:
                    if answer_lines:
                        raw_answer = "".join(answer_lines).strip()
                        current_question['answer'] = md.convert(raw_answer) if md else raw_answer
                        if md: md.reset()
                    questions.append(current_question)
                
                # Start new question
                current_question = {
                    "id": len(questions),
                    "topic": topic,
                    "question": match_q.group(1).strip(),
                    "difficulty": "Unknown",
                    "tags": [],
                    "companies": "Unknown",
                    "answer": ""
                }
                capturing_answer = False
                answer_lines = []
                continue
            
            if not current_question:
                continue
                
            # 2. Detect Metadata
            match_meta = metadata_pattern.match(line_stripped)
            if match_meta:
                current_question['difficulty'] = match_meta.group(1).strip()
                # Parse tags: `Tag1`, `Tag2` -> ["Tag1", "Tag2"]
                raw_tags = match_meta.group(2).strip()
                current_question['tags'] = [t.strip('` ') for t in raw_tags.split(',')]
                current_question['companies'] = match_meta.group(3).strip()
                continue
                
            # 3. Detect Answer Block
            if answer_start_pattern.match(line_stripped):
                capturing_answer = True
                continue
            
            # 4. Capture Answer Content
            if capturing_answer:
                # Stop capturing if we hit a horizontal rule or next header (though header logic is above)
                if line_stripped.startswith("---"):
                    capturing_answer = False
                    continue
                    
                # We need to unindent indentation to properly handle nested blocks
                # MkDocs blocks (???) are indented by 4 spaces
                if line.startswith("    "):
                    answer_lines.append(line[4:])
                else: 
                     # If the line is empty, it's fine. 
                     # If it's not indented but part of the block, it might be a format issue, 
                     # but we should preserve it to avoid losing content.
                     answer_lines.append(line)

        # Save last question
        if current_question:
            if answer_lines:
                # Join and strip common leading whitespace if any (just in case)
                raw_answer = "".join(answer_lines).strip()
                # Ensure we have newlines for tables
                current_question['answer'] = md.convert(raw_answer) if md else raw_answer
            questions.append(current_question)

    # Write to JSON
    output_path = Path(site_dir) / "assets" / "questions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    
    log.info(f"Generated questions.json with {len(questions)} questions.")
