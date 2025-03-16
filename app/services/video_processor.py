import cv2
import numpy as np
from PIL import Image
import io
import base64
from app.core.logging import logger
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from openai import AsyncOpenAI
import google.generativeai as genai
from app.core.config import settings
import asyncio
from collections import defaultdict
import tempfile
import os
from app.core.task_tracker import task_tracker
import json
import re

# Task queue to store processing results
task_queue: Dict[str, Dict] = defaultdict(dict)
#client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
client = genai.configure(api_key=settings.GEMINI_API_KEY)

async def split_video(video_content: bytes, task_id: str) -> List[bytes]:
    """
    Split video into parts based on duration.
    
    Args:
        video_content (bytes): Raw video content
        task_id (str): Unique task identifier
        
    Returns:
        List[bytes]: List of video chunks in memory
    """
    temp_input_file = None
    temp_output_file = None
    try:
        task_tracker.update_progress(task_id, "Saving video to temporary file", 6)
        # First, save the input video content to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_content)
            temp_input_file = temp_file.name
        
        task_tracker.update_progress(task_id, "Opening video file", 7)
        # Open video from the temporary file
        video = cv2.VideoCapture(temp_input_file)
        if not video.isOpened():
            raise ValueError(f"Could not open video content from {temp_input_file}")
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        task_tracker.update_progress(task_id, "Calculating video parts", 8)
        # Calculate number of parts (max 5)
        minutes = duration / 60
        num_parts = min(5, max(1, int(minutes)))
        frames_per_part = total_frames // num_parts
        
        logger.info(f"Video properties: {total_frames} frames, {fps} FPS, Duration: {duration:.2f} seconds")
        logger.info(f"Splitting into {num_parts} parts, {frames_per_part} frames per part")
        
        # Split video into parts
        video_parts = []
        progress_per_part = 7 / num_parts  # 7% progress allocated for splitting
        
        for i in range(num_parts):
            task_tracker.update_progress(task_id, f"Processing video part {i+1}/{num_parts}", 8 + (i * progress_per_part))
            
            # Create temporary file for output chunk
            with tempfile.NamedTemporaryFile(suffix=f'_part_{i}.mp4', delete=False) as out_file:
                temp_output_file = out_file.name
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                temp_output_file,
                fourcc,
                fps,
                (width, height)
            )
            
            start_frame = i * frames_per_part
            end_frame = start_frame + frames_per_part if i < num_parts - 1 else total_frames
            
            logger.info(f"Processing part {i+1}: frames {start_frame} to {end_frame}")
            
            for frame_idx in range(int(start_frame), int(end_frame)):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
                if ret:
                    writer.write(frame)
            
            writer.release()
            
            # Read the chunk back and append to list
            with open(temp_output_file, 'rb') as f:
                video_parts.append(f.read())
            
            # Clean up output file
            os.unlink(temp_output_file)
        
        task_tracker.update_progress(task_id, "Video splitting completed", 15)
        return video_parts
        
    except Exception as e:
        logger.error(f"Error in split_video: {str(e)}")
        raise
    
    finally:
        if 'video' in locals():
            video.release()
        # Clean up temporary files
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.unlink(temp_input_file)
            except Exception as e:
                logger.error(f"Error deleting temporary input file: {str(e)}")
        if temp_output_file and os.path.exists(temp_output_file):
            try:
                os.unlink(temp_output_file)
            except Exception as e:
                logger.error(f"Error deleting temporary output file: {str(e)}")

async def extract_frames(video_chunk: bytes) -> str:
    """
    Extract frames from a video chunk and create a grid visualization.
    
    Args:
        video_chunk (bytes): Video chunk content
        
    Returns:
        str: Base64 encoded grid image
    """
    temp_file = None
    try:
        # Save video chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp:
            temp.write(video_chunk)
            temp_file = temp.name
        
        # Open video from temporary file
        video = cv2.VideoCapture(temp_file)
        if not video.isOpened():
            raise ValueError(f"Could not open video chunk from {temp_file}")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("Video chunk contains no frames")
            
        interval = max(1, total_frames // 16)
        logger.info(f"Extracting frames from chunk: {total_frames} total frames, interval {interval}")
        
        frames = []
        for i in range(16):
            frame_pos = min(i * interval, total_frames - 1)  # Ensure we don't exceed total frames
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = video.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                logger.warning(f"Failed to read frame at position {frame_pos}")
        
        if not frames:
            logger.warning("No frames were extracted from the video chunk")
            return None
            
        logger.info(f"Successfully extracted {len(frames)} frames")
        
        # Create grid image
        grid = Image.new('RGB', (frames[0].shape[1] * 4, frames[0].shape[0] * 4))
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            x_position = (i % 4) * frames[0].shape[1]
            y_position = (i // 4) * frames[0].shape[0]
            grid.paste(img, (x_position, y_position))
        
        # Convert to base64
        buffer = io.BytesIO()
        grid.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error in extract_frames: {str(e)}")
        return None
        
    finally:
        if 'video' in locals():
            video.release()
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

async def check_content_moderation(base64_images: List[str]) -> Tuple[bool, List[str]]:
    """
    Ultra-strict content moderation using OpenAI's moderation API.
    Extremely conservative thresholds for all categories.
    """
    try:
        all_warnings = []
        is_safe = True
        
        # Ultra-strict thresholds for different categories
        thresholds = {
            'sexual': 0.25,            # 25%
            'sexual/minors': 0.25,   # 25%
            'violence': 0.25,         # 25%
            'violence/graphic': 0.25,  # 25%
            'harassment': 0.25,       # 25%
            'harassment/threatening': 0.25,  # 25%
            'hate': 0.25,            # 25%
            'hate/threatening': 0.25,  # 25%
            'self-harm': 0.25,        # 25%
            'self-harm/intent': 0.25,  # 25%
            'self-harm/instructions': 0.25,  # 25%
            'illicit': 0.25,         # 25%
            'illicit/violent': 0.25    # 25%
        }
        
        # Process each image individually
        for idx, base64_image in enumerate(base64_images):

            #Changed as per GEMINI API
            image_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/png", "data": image_data}

            prompt = """
            Analyze the following image for the presence of:
            - Sexual content
            - Sexual content involving minors
            - Violence
            - Graphic violence
            - Harassment
            - Threatening harassment
            - Hate speech
            - Threatening hate speech
            - Self-harm
            - Intentional self-harm
            - Instructions for self-harm
            - Illicit activities
            - Violent illicit activities

            Provide a score (0-1) for each category, where 1 indicates high confidence.
            Also, indicate if the content is flagged as unsafe (yes/no).
            Respond in a structured format, like:
            Sexual: 0.8, Flagged: yes
            Violence: 0.1, Flagged: no
            ...
            """

            # response = await client.moderations.create(
            #     # model="omni-moderation-latest",
            #     #model=genai.GenerativeModel("gemini-pro-vision"),
            #     model=genai.GenerativeModel("gemini-1.5-flash"),
            #     input=[{
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/png;base64,{base64_image}"
            #         }
            #     }]
            # )

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt, image_part])

            if not response or not response.text:
                logger.error("Error: Empty or invalid response from Gemini API")
                return False, ["CRITICAL RISK - Error processing content moderation"]

            response_text = response.text.strip()
            logger.info(f"Moderation Response: {response_text}")

            #print(response_text)

            # Parse Gemini response
            result = parse_moderation_response(response_text)

            #print(result)
            # Check if response was parsed correctly
            if not result:
                logger.warning("No categories detected in response, possible parsing issue.")
                continue

            # Check if content is flagged by any category
            if any(res.get('flagged', False) for res in result.values()):
                is_safe = False
                all_warnings.append("SYSTEM FLAG - Content flagged by moderation system")
            
            # Check all categories with their specific thresholds
            """for category in thresholds.keys():
                # Get score safely using getattr
                score = getattr(result.category_scores, category.replace('/', '_'), 0.0)"""
            #print("HERE 1")
            # Check all categories with their specific thresholds
            for category,threshold in thresholds.items():
                normalized_category = category.replace('/', ' ').lower()
                #print("HERE 2")
                # Ensure the category exists in result, else use default score of 0.0
                score = result.get(normalized_category, {}).get('score', 0.0)
                #print("HERE 3")
                # Validate score type
                if not isinstance(score, (int, float)):
                    continue # Skip invalid scores

                display_category = category.replace('/', ' - ').title()

                #print(display_category)

                if score >= threshold:
                    is_safe = False
                    if score > 0.7:
                        severity = "CRITICAL"
                    elif score > 0.4:
                        severity = "HIGH"
                    elif score > 0.2:
                        severity = "MEDIUM"
                    else:
                        # pass
                        severity = "LOW" # In low if confidence is below 25% then it is safe
                    all_warnings.append(f"{severity} RISK - {display_category} detected (confidence: {score:.1%})")

                # Check applied input types for additional context (e.g., image)
                if normalized_category in result: # Ensure category exists in result
                    if "image" in normalized_category:  # Example condition for image-based content
                        if score >= thresholds[category]:
                            display_category = category.replace('/', ' - ').title()
                            all_warnings.append(f"IMAGE SPECIFIC - {display_category} detected in visual content")

                #print("Now here")

            # Check applied input types for additional context
            """for category in thresholds.keys():
                category_key = category.replace('/', '_')
                types = getattr(result.category_applied_input_types, category_key, [])
                score = getattr(result.category_scores, category_key, 0.0)
                
                if "image" in types and score >= thresholds[category]:
                    display_category = category.replace('/', ' - ').title()
                    all_warnings.append(f"IMAGE SPECIFIC - {display_category} detected in visual content")"""

            # Check applied input types for additional context (e.g., image)
            """for category in thresholds.keys():
                normalized_category = category.replace('/', ' ').lower()

                if normalized_category in result:  # Ensure category exists in result
                    # Extract the score for the category safely (default to 0.0 if not present)
                    score = result[normalized_category].get('score', 0.0)"""

        #print("Now before duplicates")
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_warnings = []
        for warning in all_warnings:
            if warning not in seen:
                seen.add(warning)
                filtered_warnings.append(warning)
        
        # Sort warnings by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "SYSTEM FLAG": 4}
        filtered_warnings.sort(key=lambda x: severity_order.get(x.split()[0], 5))
        
        # Add summary warnings for high-risk content
        if not is_safe:
            summary_warnings = []
            if any("CRITICAL" in w for w in filtered_warnings):
                summary_warnings.append("CRITICAL RISK - Severe content violations detected")
            if any("HIGH" in w for w in filtered_warnings):
                summary_warnings.append("HIGH RISK - Significant content concerns identified")
            filtered_warnings = summary_warnings + filtered_warnings
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filtered_warnings.insert(0, f"Content Moderation Timestamp: {timestamp}")
        
        # Log results
        logger.info(f"Content moderation complete - Safe: {is_safe}, Warnings: {len(filtered_warnings)}")
        for warning in filtered_warnings:
            logger.warning(f"Content Warning: {warning}")
        
        return is_safe, filtered_warnings
    
    except Exception as e:
        logger.error(f"Error in content moderation: {str(e)}")
        return False, ["CRITICAL RISK - Error in content moderation system"]


# Customized code for Parsing response from moderation
def parse_moderation_response(response_text: str) -> dict:
    """
    Parses the Gemini API response text to extract category scores and flags.
    """
    result = {}

    # Regex to match "Category: Score, Flagged: yes/no"
    pattern = r"\*\*(.*?)\*\*:\s*([\d.]+).*?Flagged:\s*(yes|no)"
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    print(matches)

    if matches:
        for category, score_str, flagged_str in matches:
            category = category.strip().lower()
            score = float(score_str)
            flagged = flagged_str.lower() == "yes"
            result[category] = {"score": score, "flagged": flagged}
        return result

    # If regex fails, use alternative parsing
    lines = response_text.split("\n")
    for i in range(0, len(lines), 2):  # Process two lines at a time
        if i + 1 < len(lines) and "Flagged:" in lines[i + 1]:
            category = lines[i].strip().replace("**", "").lower()
            score_match = re.search(r"([\d.]+)", lines[i])
            if score_match:
                score = float(score_match.group())
                flagged = "yes" in lines[i + 1].lower()
                result[category] = {"score": score, "flagged": flagged}

    return result


async def analyze_grid_images(base64_images: List[str], task_id: str = None) -> List[str]:
    """
    Analyze grid images using GPT-4 vision model.
    """
    try:
        descriptions = []
        total_images = len(base64_images)
        
        for idx, base64_image in enumerate(base64_images, 1):
            if task_id:
                progress = 65 + (idx * 5 / total_images)
                task_tracker.update_progress(task_id, f"Analyzing grid image {idx}/{total_images}", progress)
            
            try:
                """response = await client.chat.completions.create(
                    #model="gpt-4",
                    model = genai.GenerativeModel("gemini-pro-vision"),
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a video frame analysis expert. Describe the key visual elements, actions, and details in this frame grid."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this grid of video frames. Focus on: main subjects, actions, visual elements, text overlays, scene composition, and any notable details."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                
                description = response.choices[0].message.content.strip()
                descriptions.append(description)"""

                #Changed as per GEMINI API

                # Decode base64 image to raw bytes
                image_data = base64.b64decode(base64_image)

                # Initialize Gemini model
                #model = genai.GenerativeModel("gemini-pro-vision") #deprecated
                model = genai.GenerativeModel("gemini-1.5-flash")

                # Combine system and user instructions into the prompt
                prompt = [
                    "You are a video frame analysis expert. Describe the key visual elements, actions, and details in this frame grid.",
                    {
                        "mime_type": "image/png",  # Ensure correct MIME type
                        "data": image_data
                    },
                    "Analyze this grid of video frames. Focus on: main subjects, actions, visual elements, text overlays, scene composition, and any notable details."
                ]

                # Send the combined prompt to Gemini API
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(max_output_tokens=500)
                )

                # Extract and store the description
                descriptions.append(response.text.strip())


            except Exception as e:
                logger.error(f"Error analyzing grid image {idx}: {str(e)}")
                descriptions.append(f"Error analyzing frame grid {idx}")
        
        return descriptions
        
    except Exception as e:
        logger.error(f"Error in grid image analysis: {str(e)}")
        return ["Error analyzing frame grids"]

async def process_video(video_content: bytes, task_id: str) -> Tuple[bool, List[str], List[str]]:
    """
    Main video processing function that coordinates the entire workflow.
    """
    try:
        task_tracker.update_progress(task_id, "Starting video processing", 5)
        
        # Split video into parts
        video_parts = await split_video(video_content, task_id)
        task_tracker.update_progress(task_id, "Video split completed", 15)
        
        # Process each part in parallel
        tasks = [extract_frames(part) for part in video_parts]
        base64_grids = await asyncio.gather(*tasks)
        task_tracker.update_progress(task_id, "Frame extraction completed", 25)
        
        # Store grids in task queue
        task_queue[task_id]['grids'] = base64_grids
        
        # Filter out None values and check content moderation for all grids in one call
        valid_grids = [grid for grid in base64_grids if grid is not None]
        if valid_grids:
            task_tracker.update_progress(task_id, "Starting content moderation", 30)
            is_safe, warnings = await check_content_moderation(valid_grids)
            task_tracker.update_progress(task_id, "Content moderation completed", 35)
        else:
            is_safe, warnings = False, ["No valid frames extracted"]
            valid_grids = []
        
        print(f"\n{'='*30}\nIs Safe: {is_safe}\n{'='*30}")
        print(f"\n{'='*30}\nWarnings: {warnings}\n{'='*30}")
        
        # Store results in task queue
        task_queue[task_id]['is_safe'] = is_safe
        task_queue[task_id]['warnings'] = warnings
        
        return is_safe, warnings, valid_grids
    
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
        task_queue[task_id]['error'] = str(e)
        return False, [f"Processing error: {str(e)}"], []