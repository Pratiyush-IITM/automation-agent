import os
import subprocess
import json
import sqlite3
from app.llm import parse_task, call_llm
from app.utils import safe_path
import cv2
import pytesseract
import numpy as np
from sentence_transformers import SentenceTransformer, util

DATA_DIR = "/data"

def execute_task(task_description: str):
    """Parses and executes the task safely."""
    parsed_task = parse_task(task_description)
    
    if "get recent logs" in parsed_task:
        get_recent_logs()

    elif "index markdown" in parsed_task:
        index_markdown()

    elif "extract email sender" in parsed_task:
        extract_email_sender()

    elif "extract credit card" in parsed_task:
        extract_credit_card()

    elif "find similar comments" in parsed_task:
        find_similar_comments()

    elif "calculate gold ticket sales" in parsed_task:
        calculate_gold_ticket_sales()

    else:
        raise ValueError(f"Task '{task_description}' not recognized")
    
    return f"Task '{task_description}' completed."

def get_recent_logs():
    """A5: Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt"""
    log_dir = safe_path(f"{DATA_DIR}/logs")
    output_file = safe_path(f"{DATA_DIR}/logs-recent.txt")

    log_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".log")],
        key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
        reverse=True
    )[:10]

    with open(output_file, "w") as out:
        for log_file in log_files:
            with open(os.path.join(log_dir, log_file)) as f:
                first_line = f.readline().strip()
                out.write(first_line + "\n")

def index_markdown():
    """A6: Create an index of Markdown (.md) files with their first H1 title"""
    docs_dir = safe_path(f"{DATA_DIR}/docs")
    index_file = safe_path(f"{DATA_DIR}/docs/index.json")

    index = {}
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file_path.replace(f"{docs_dir}/", "")] = line[2:].strip()
                            break

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

def extract_email_sender():
    """A7: Extract the sender’s email address from /data/email.txt"""
    input_file = safe_path(f"{DATA_DIR}/email.txt")
    output_file = safe_path(f"{DATA_DIR}/email-sender.txt")

    with open(input_file, "r") as f:
        email_content = f.read()

    sender_email = call_llm(f"Extract the sender’s email from this message:\n\n{email_content}")

    with open(output_file, "w") as f:
        f.write(sender_email.strip())

def extract_credit_card():
    """A8: Extract the credit card number from an image"""
    image_file = safe_path(f"{DATA_DIR}/credit-card.png")
    output_file = safe_path(f"{DATA_DIR}/credit-card.txt")

    img = cv2.imread(image_file, 0)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(img_bin)

    card_number = "".join(filter(str.isdigit, text))

    with open(output_file, "w") as f:
        f.write(card_number)

def find_similar_comments():
    """A9: Find the most similar pair of comments based on embeddings"""
    input_file = safe_path(f"{DATA_DIR}/comments.txt")
    output_file = safe_path(f"{DATA_DIR}/comments-similar.txt")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(input_file, "r") as f:
        comments = [line.strip() for line in f.readlines()]

    embeddings = model.encode(comments, convert_to_tensor=True)
    best_score = 0
    best_pair = ("", "")

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            score = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if score > best_score:
                best_score = score
                best_pair = (comments[i], comments[j])

    with open(output_file, "w") as f:
        f.write(best_pair[0] + "\n" + best_pair[1])

def calculate_gold_ticket_sales():
    """A10: Compute total sales for Gold tickets"""
    db_file = safe_path(f"{DATA_DIR}/ticket-sales.db")
    output_file = safe_path(f"{DATA_DIR}/ticket-sales-gold.txt")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type='Gold'")
    total_sales = cursor.fetchone()[0] or 0

    with open(output_file, "w") as f:
        f.write(str(total_sales))

    conn.close()
