from io import BytesIO
from fastapi import APIRouter, Body, UploadFile, File, BackgroundTasks, Header
from app.services.video_processor import process_video
from app.services.audio_processor import process_audio
from app.services.gpt_service import generate_description
from app.services.keyword_extractor import extract_video_metadata
from app.core.logging import logger
from fastapi.param_functions import Query
from app.core.config import settings
from app.core.task_tracker import task_tracker
from typing import Optional
import uuid
import asyncio
import requests
import os
import time
import json

MAX_RETRIES = 3
RETRY_DELAY = 10

EMPOWERVERSE_API_KEY = settings.EMPOWERVERSE_API_KEY
WEMOTIONS_API_KEY = settings.WEMOTIONS_API_KEY
EMPOWERVERSE_API_PATH = settings.EMPOWERVERSE_API_PATH
WEMOTIONS_API_PATH = settings.WEMOTIONS_API_PATH
VIDEO_DESCRIPTION_KEY = settings.VIDEO_DESCRIPTION_KEY

router = APIRouter()

analysis_results = {}

async def analyze_video_task(video_content: bytes, video_filename: str, task_id: str, app_name: str, identifier: Optional[str] = None, is_christian_content: Optional[bool] = False):
    audio_result = None
    try:
        task_tracker.start_task(task_id)
        current_progress = 0
        
        # Run process_video and process_audio in parallel
        task_tracker.update_progress(task_id, "Starting parallel processing", current_progress)
        video_task = asyncio.create_task(process_video(video_content, task_id))
        audio_task = asyncio.create_task(process_audio(video_content, task_id))
        
        # Wait for both tasks to complete and handle their results
        video_result, audio_result = await asyncio.gather(video_task, audio_task)
        current_progress = 20
        task_tracker.update_progress(task_id, "Parallel processing completed", current_progress)
        
        # Unpack video processing results
        is_safe, content_warnings, base64_grids = video_result
        current_progress = 30
        task_tracker.update_progress(task_id, "Video processing results unpacked", current_progress)
        
        # Extract audio transcription from audio result
        audio_transcription = ''
        try:
            if isinstance(audio_result, (list, tuple)) and audio_result:
                first_result = audio_result[0]
                if isinstance(first_result, dict):
                    audio_transcription = first_result.get('text', '')
                elif hasattr(first_result, 'text'):
                    audio_transcription = first_result.text
        except Exception as e:
            logger.error(f"Error extracting audio transcription: {str(e)}")
            audio_transcription = ''
        
        current_progress = 40
        task_tracker.update_progress(task_id, "Audio transcription extracted", current_progress)
        
        # Skip further processing if no valid frames were extracted
        if not base64_grids:
            logger.warning("No valid frames were extracted from the video")
            task_tracker.complete_task(task_id, "error")
            analysis_results[task_id] = {
                "status": "error",
                "message": "No valid frames could be extracted from the video"
            }
            return
        
        # Generate comprehensive description
        task_tracker.update_progress(task_id, "Generating description", current_progress)
        description = await generate_description(base64_grids, audio_transcription, task_id)
        current_progress = 60
        task_tracker.update_progress(task_id, "Description generated", current_progress)
        
        # Extract metadata
        task_tracker.update_progress(task_id, "Extracting metadata", current_progress)
        metadata = await extract_video_metadata(description, is_christian_content, task_id)
        current_progress = 80
        task_tracker.update_progress(task_id, "Metadata extracted", current_progress)
        
        # Create the result dictionary with required fields
        result = {
            "status": "completed",
            "description": description or "",
            "is_safe": is_safe,
            "content_warnings": content_warnings or [],
            "keywords": metadata.get("keywords", []),
            "is_face_exist": metadata.get("is_face_exist", False)
        }

        # Add OpenAI-provided fields only if they are present
        if "topics" in metadata:
            result["topics"] = metadata["topics"]

        if "entities" in metadata:
            result["entities"] = metadata["entities"]

        if "actions" in metadata:
            result["actions"] = metadata["actions"]

        if "emotions" in metadata:
            result["emotions"] = metadata["emotions"]

        if "visual_elements" in metadata:
            result["visual_elements"] = metadata["visual_elements"]

        if "audio_elements" in metadata:
            result["audio_elements"] = metadata["audio_elements"]

        if "genre" in metadata:
            result["genre"] = metadata["genre"]

        if "target_audience" in metadata:
            result["target_audience"] = metadata["target_audience"]

        if "quality_indicators" in metadata:
            result["quality_indicators"] = metadata["quality_indicators"]

        if "unique_identifiers" in metadata:
            result["unique_identifiers"] = metadata["unique_identifiers"]

        if "person_identity" in metadata:
            result["person_identity"] = metadata["person_identity"]

        if "other_person_identity" in metadata:
            result["other_person_identity"] = metadata["other_person_identity"]

        if "psychological_personality" in metadata:
            result["psychological_personality"] = metadata["psychological_personality"]

        if "no_of_person_in_video" in metadata:
            value = metadata["no_of_person_in_video"]
            if isinstance(value, str):
                result["no_of_person_in_video"] = int(value) if value.isdigit() else 0
            else:
                result["no_of_person_in_video"] = value


        # Add Christian content analysis if present
        if is_christian_content:
            christianContent = metadata.get("is_christian_content", {})
            result["is_christian_content"] = {
                "is_christian": christianContent.get("is_christian", False),
                "confidence_score": christianContent.get("confidence_score"),
                "indicators": christianContent.get("indicators")
            }
        
        print(f"\n{'#'*30}\nResult: {result}\n{'#'*30}")
        
        analysis_results[task_id] = result
        current_progress = 90
        task_tracker.update_progress(task_id, "Results compiled", current_progress)

        # Handle callback if identifier is provided
        if identifier:
            task_tracker.update_progress(task_id, "Sending callback", current_progress)
            """print(f"Sending callback to: {EMPOWERVERSE_API_PATH}/post/summary")
            # api_url = f"{EMPOWERVERSE_API_PATH if app_name == 'empowerverse' else WEMOTIONS_API_PATH}/post/summary"
            api_url = ""
            if app_name == 'empowerverse':
                api_url = f"{EMPOWERVERSE_API_PATH}/post/summary" 
            elif app_name == 'wemotions':
                f"{WEMOTIONS_API_PATH}/post/summary"
            """
            
            # api_url = "http://localhost:8000/post/summary"
            """api_url = " http://127.0.0.1:8000/post/summary"
            payload = {
                "key": VIDEO_DESCRIPTION_KEY,
                "identifier": identifier,
                **result
            }
            
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                logger.info("Data sent to PHP API successfully.")
            else:
                logger.error(f"Failed to send data to PHP API: {response.content}")"""
        
        current_progress = 100
        task_tracker.update_progress(task_id, "Task completed", current_progress)
        task_tracker.complete_task(task_id)

    except Exception as e:
        logger.error(f"Error during video analysis: {str(e)}")
        task_tracker.complete_task(task_id, "error")
        analysis_results[task_id] = {"status": "error", "message": str(e)}
    finally:
        # Clean up any remaining audio files if task status is either error or completed
        task_data = task_tracker.tasks.get(task_id, {})
        task_status = task_data.get("status")
        
        if task_status in ["error", "completed"]:
            if audio_result and isinstance(audio_result, tuple) and len(audio_result) > 1:
                audio_filename = audio_result[1]
                if audio_filename and os.path.exists(audio_filename):
                    try:
                        os.unlink(audio_filename)
                        logger.info(f"Cleaned up audio file: {audio_filename} (Task status: {task_status})")
                    except Exception as e:
                        logger.error(f"Error cleaning up audio file: {str(e)}")
                else:
                    logger.info(f"No audio file to clean up for task {task_id} (Task status: {task_status})")
        else:
            logger.info(f"Skipping audio cleanup for task {task_id} (Task status: {task_status})")

@router.post("/analyze_video")
async def analyze_video(background_tasks: BackgroundTasks, app_name: str, video: UploadFile = File(None), file_url: Optional[str] = Query(None), identifier: Optional[str] = Query(None), is_christian_content: Optional[bool] = Query(False)):
    try:
        task_id = str(uuid.uuid4())
        progress_bar = 0

        if file_url:
            for attempt in range(MAX_RETRIES):
                response = requests.get(file_url)
                if response.status_code == 200:
                    response.raise_for_status()
                    file_content = response.content
                    filename = os.path.basename(file_url)

                    progress_bar += 1
                    background_tasks.add_task(analyze_video_task, file_content, filename, task_id, app_name, identifier, is_christian_content)
                    break

                print(f"Attempt {attempt + 1}: Result not ready. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
        
        elif video:
            video_content = await video.read()
            progress_bar += 1
            background_tasks.add_task(analyze_video_task, video_content, video.filename, task_id, identifier, is_christian_content)
        
        return {
            "message": "Video analysis started.",
            "task_id": task_id
        }
    
    except requests.RequestException as e:
        logger.error(f"Error during video analysis: {str(e)}")
        return {"error": "Failed to process video"}    

@router.get("/analysis_result/{task_id}")
async def get_analysis_result(task_id: str):
    result = analysis_results.get(task_id)
    print(result)
    if result is None:
        # Get progress from task tracker
        task_data = task_tracker.tasks.get(task_id)
        if task_data:
            return {
                "status": "pending",
                "progress": task_data["current_progress"],
                "current_step": list(task_data["steps"].keys())[-1] if task_data["steps"] else None
            }
        return {"status": "pending", "progress": 0}
    return result

@router.post("/share_url")
#async def share_url(background_tasks: BackgroundTasks, flic_token: str = Header(...), data: dict = Body(...))
async def share_url(background_tasks: BackgroundTasks,  data: dict = Body(...)):
    url = data.get('url')
    identifier = data.get('identifier'),
    is_christian_content = data.get('is_christian_content', False)

    app_name = "test"
     # Check if flic_token is valid for either API key
    """if flic_token == EMPOWERVERSE_API_KEY:
        app_name = "empowerverse"
    elif flic_token == WEMOTIONS_API_KEY:
        app_name = "wemotions"
    else:
        return {
            "status": "error",
            "message": "Invalid Flic_Token"
        }"""

    if not url or not identifier:
        return {
            "status": "error",
            "message": "url and identifier are required fields"
        }

    #removed app_name here as not validating flic_token
    background_tasks.add_task(analyze_video, background_tasks, app_name, file_url=url, identifier=identifier, is_christian_content=is_christian_content)
    return {
        "status": "success",
        "message": "URL processed successfully, video processing in queue..."
    }
