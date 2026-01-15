from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
import pandas as pd
import csv
import os
import requests
import re


local_model_path = ".\qwen_path"
user_item_file = r".\Data\baby\new_baby_user_item_matrix.npy"
item_info_file = r".\Data\baby\combined_output.csv"
output_csv = r".\Data\baby\user_portraits.csv"

def get_last_processed_user(output_csv):
    last_user = 0
    if os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)

                lines = f.readlines()
                if not lines:
                    return 0

                for line in reversed(lines[-10:]):
                    row = line.strip().split(',')
                    if len(row) >= 2:
                        user_id = row[0].strip()
                        if user_id.isdigit():
                            last_user = int(user_id)
                            break
        except Exception as e:
            print(f"Error in: {str(e)}")
    return last_user


def choose_restart_mode():
    if os.path.exists(output_csv):
        last_user = get_last_processed_user(output_csv)

        print(f"Output file detected: {output_csv}")
        print("Please select processing mode:")
        print("1 - Restart (overwrite existing files)")
        print("2 - Resume interrupted download (continue processing from the last user)")

        while True:
            choice = input("Please enter your option (1 or 2):")
            if choice == '1':
                return 'restart', 0
            elif choice == '2':
                return 'resume', last_user
            else:
                print("Invalid input, please select again.")
    else:
        print("No output file is found, processing will restart.")
        return 'restart', 0

def is_image_accessible(url, timeout=2):
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            return content_type.startswith('image/')
        return False
    except (requests.RequestException, ValueError):
        return False

mode, resume_from = choose_restart_mode()
print(f"Selected mode: {mode.upper()}")

if mode == 'restart':
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['userID', 'userPortrait'])
    print("The output file has been reset and will be processed from scratch.")

user_item_data = np.load(user_item_file, allow_pickle=True).item()

item_df = pd.read_csv(item_info_file)
item_info_dict = item_df.set_index('itemID')[['imUrl', 'text']].apply(tuple, axis=1).to_dict()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=local_model_path,
    torch_dtype=torch.float,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(local_model_path)

all_user_ids = sorted([int(uid) for uid in user_item_data.keys()])
total_users = len(all_user_ids)
start_index = 0

if mode == 'resume':
    if resume_from > 0:
        try:
            start_index = all_user_ids.index(resume_from) + 1
            print(f"Processing will begin after the user ID {resume_from} (location {start_index}/{total_users}).")
        except ValueError:
            print(f"Warning: User ID {resume_from} is not in the dataset. It will be processed from the beginning.")
    else:
        print("No valid resume point is found, processing will start from the beginning.")

processed_count = 0
to_process = total_users - start_index

for i in range(start_index, total_users):
    user_id = all_user_ids[i]

    items = user_item_data.get(user_id, [])

    processed_count += 1
    progress = f"({i + 1}/{total_users}, Processed: {processed_count}/{to_process})"
    print(f"\n{progress} processes user {user_id}...")

    interacted_items = items[0][:20] if items and items[0] else []
    print(f"This user has {len(interacted_items)} interactive items.")

    if not interacted_items:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, "No interacted items"])
        print(f"User {user_id} has not interacted with any items; this has been skipped.")
        continue

    message_content = [
        {"type": "text", "text": "Generate a user profile based on these purchased items:"}
    ]

    valid_item_count = 0
    for _, item_id in enumerate(interacted_items):
        if item_id in item_info_dict:
            imUrl, text = item_info_dict[item_id]
            img_added = False
            text_added = False

            if imUrl and pd.notna(imUrl) and imUrl.strip().lower() != "none":
                clean_url = imUrl.strip()
                if is_image_accessible(clean_url):
                    try:
                        message_content.append({"type": "image", "image": clean_url})
                        img_added = True
                    except Exception as e:
                        print(f"Image added failed.: {clean_url} - {str(e)}")
                else:
                    print(f"Image inaccessible: {clean_url}")

            if text and pd.notna(text):
                message_content.append({
                    "type": "text",
                    "text": f"Item {valid_item_count + 1}: {text}"[:500]
                })
                text_added = True

            if img_added or text_added:
                valid_item_count += 1

    if valid_item_count == 0:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, "No valid item info"])
        print(f"User {user_id} has no valid item information and has been skipped.")
        continue

    message_content.append({
        "type": "text",
        "text": "Synthesize visual and textual features to generate a one-sentence user profile."
    })

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional user profiling analyst who generates user profiles based on purchased items' images and text descriptions.\n"
                "## Task Requirements:\n"
                "1. Synthesize visual features and text descriptions to identify user characteristics\n"
                "2. Profile dimensions must include: age range, gender, spending level, lifestyle preferences, core needs\n"
                "3. Output results in **ONE sentence** (max 40 characters)\n"
                "4. Base conclusions on product features, avoid subjective assumptions\n\n"
                "## Analysis Framework:\n"
                "1. Visual Analysis: Product category/colors/design style/usage scenarios\n"
                "2. Text Analysis: Function descriptions/price range/target users\n"
                "3. Cross-product Analysis: Identify common patterns across items"
            )
        },
        {"role": "user", "content": message_content}
    ]

    try:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, output_text])

        print(f"Successfully generated user profile {user_id}: {output_text.strip()}")

    except Exception as e:
        error_msg = f"Error processing: {str(e)[:200]}"
        print(error_msg)
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, error_msg])

print("\nDone")