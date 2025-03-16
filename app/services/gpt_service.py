import base64
from openai import AsyncOpenAI
import google.generativeai as genai
from PIL import Image
import io
from app.core.task_tracker import task_tracker
from app.core.config import settings
from app.core.logging import logger
from typing import List


#client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
client = genai.configure(api_key=settings.GEMINI_API_KEY)

async def analyze_grid_images(base64_images: List[str], task_id: str = None) -> List[str]:
    """
    Analyze multiple grid images without audio and return their descriptions.
    
    Args:
        base64_images (List[str]): List of base64 encoded grid images
        task_id (str, optional): Task identifier for progress tracking
        
    Returns:
        List[str]: List of descriptions for each grid
    """
    try:
        descriptions = []
        total_images = len(base64_images)
        #model = genai.GenerativeModel("gemini-pro-vision") #deprecated
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        for idx, base64_image in enumerate(base64_images, 1):
            if task_id:
                progress = int(65 + (idx / total_images * 5))  # Progress from 65% to 70%
                task_tracker.update_progress(task_id, f"Analyzing grid image {idx}/{total_images}", progress)
                
            prompt = """
            Analyze this series of video frames with particular attention to Christian themes and NSFW content:

            Provide a comprehensive description focusing on:
            1. The speaker's actions and expressions
            2. Any text overlays or icons and their significance
            3. Visual elements and their significance
            4. The overall theme and message visible in these frames
            5. Number of human faces visible
            6. Gender identification of visible individuals
            7. Personality traits and demeanor of main individuals
            8. Notable interactions or expressions
            9. Visual progression and scene changes
            10. Any identifiable individuals or notable features
            11. Religious or spiritual elements present (crosses, churches, religious symbols, etc.)
            12. Any visible scripture references or biblical content
            13. Signs of worship, prayer, or religious activities
            14. Evidence of Christian values (love, service, humility, etc.)
            15. Any religious gatherings or community events

            Focus on visual analysis only. Describe the progression naturally without mentioning grid layout.
            Pay special attention to elements that indicate Christian content or messaging.
            """

            """response = await client.chat.completions.create(
                #model="gpt-4o",
                model = genai.GenerativeModel("gemini-pro-vision"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
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
            )"""
            #Changes as per Gemini API
            image_data = base64.b64decode(base64_image)
            #image_part = genai.Part.from_data(mime_type="image/png", data=image_data)

            #image = Image.open(io.BytesIO(image_data))
            #image_part = genai.upload_image(image, mime_type="image/png")
            response = model.generate_content(
                        [prompt,
                         {
                             "mime_type": "image/png",  # Specify image format
                             "data": image_data  # Pass the raw image bytes inside a dictionary
                         }
                         ],
                        generation_config=genai.GenerationConfig(max_output_tokens=1500)
            )
            descriptions.append(response.text.strip())
        
        return descriptions
    except Exception as e:
        error_msg = f"Error in analyzing grid images: {str(e)}"
        logger.error(error_msg)
        if task_id:
            task_tracker.update_progress(task_id, f"Error: {error_msg}", 70)
        return [error_msg]

async def generate_description(base64_images: List[str], audio_transcription: str = None, task_id: str = None) -> str:
    """
    Generate a comprehensive video description combining multiple grid analyses and audio transcription.
    
    Args:
        base64_images (List[str]): List of base64 encoded grid images
        audio_transcription (str, optional): Audio transcription text
        task_id (str, optional): Task identifier for progress tracking
        
    Returns:
        str: Combined comprehensive description
    """
    try:
        # First, analyze all grid images
        if task_id:
            task_tracker.update_progress(task_id, "Starting grid analysis", 65)
            
        grid_descriptions = await analyze_grid_images(base64_images, task_id)
        
        # Prepare the final analysis prompt
        if task_id:
            task_tracker.update_progress(task_id, "Generating final description", 70)
            
        combined_descriptions = "\n\n".join(grid_descriptions)
        final_prompt = f"""
        Based on the following video segment descriptions and audio transcription, provide a comprehensive analysis:

        Video Segments Analysis:
        {combined_descriptions}

        Audio Transcription:
        {audio_transcription if audio_transcription else "No audio transcription available."}

        Provide a unified description that includes:
        1. The speaker's actions and expressions
        2. Any text overlays or icons and their significance
        3. How the visuals complement or illustrate the audio content
        4. The overall theme and message of the video
        5. No of human being face visible in the video
        6. Count the number of visible human faces in the video
        7. Assess the personality traits of the main individual featured, including insights into their demeanor and engagement
        8. Identify the gender of the main speaker at the beginning of the video, as well as the genders of all other individuals present
        9. If possible, provide names or identities of other individuals featured in the video
        10. Determine whether the video contains visible faces throughout its duration
        11. Calculate the exact duration of the audio and the total length of the video based on automated processing
        12. Describe the speaker's actions, expressions, and any notable interactions
        13. Speaker Identification: If there are multiple speakers, indicate who is speaking at each point
        14. Use paragraph breaks for different speakers or topics to enhance readability

        Additionally, analyze the Christian content aspects:
        15. Identify any religious symbols, scripture references, or biblical content
        16. Note any expressions of faith, prayer, or worship
        17. Assess if the content aligns with Christian values (love, service, humility, etc.)
        18. Identify any religious gatherings or community events
        19. Evaluate if the content promotes Christian teachings or messages
        20. Determine if the content reflects Christian social media principles:
            - Authenticity over perfection
            - Focus on serving others rather than self-promotion
            - Building genuine connections and community
            - Sharing truth with grace
            - Using platform for Kingdom purposes
            - Maintaining depth over superficiality
            - Promoting unity and understanding
            - Demonstrating Christian values in presentation and message

        Based on these aspects, include a clear assessment of whether this can be classified as Christian content.
        Do not mention anything about a grid or layout of the images.
        Provide a natural, flowing narrative that combines all these elements into a coherent analysis.
        """

        # Generate final combined description
        """response = await client.chat.completions.create(
            #model="gpt-4",
            model = genai.GenerativeModel("gemini-pro-vision"),
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            max_tokens=1500
        )"""
        #Changed as per Gemini API
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
                    final_prompt,
                    generation_config=genai.GenerationConfig(max_output_tokens=1500)
                                          )
        
        if task_id:
            task_tracker.update_progress(task_id, "Description generation completed", 75)
            
        return response.text.strip()
    except Exception as e:
        error_msg = f"Error in generate_description: {str(e)}"
        logger.error(error_msg)
        if task_id:
            task_tracker.update_progress(task_id, f"Error: {error_msg}", 75)
        return error_msg
    
