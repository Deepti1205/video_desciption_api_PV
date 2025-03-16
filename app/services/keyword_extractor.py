from openai import AsyncOpenAI
import google.generativeai as genai
from app.core import task_tracker
from app.core.config import settings
from app.core.logging import logger
import json
import re

#client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
client = genai.configure(api_key=settings.GEMINI_API_KEY)


async def extract_video_metadata(description: str, is_christian_content: bool = False, task_id: str = None) -> dict:
    """
    Extract metadata from video description using GPT-4.
    
    Args:
        description (str): Video description
        is_christian_content (bool): Whether to analyze for Christian content
        task_id (str): Task identifier for progress tracking
        
    Returns:
        dict: Extracted metadata with all required fields
    """
    try:
        # Refined prompt for structured extraction
        prompt = f"""
        You are an expert content analyst. 
        Analyze the following video description and extract metadata in the exact JSON structure provided below:
        
        Video Description:
        {description}

        Extracted Metadata:
        {{
            "description": "{description}",  // Original video description
            "keywords": [
                {{"keyword":"string","weight":int}}  // Extract 10 most relevant keywords with weights (1-10) 
                                                     
            ],
            "topics": ["string"],  // List at least 3 key topics discussed
            "entities": ["string"],  // Mentioned people, organizations, or objects
            "actions": ["string"],  // Key actions described
            "emotions": ["string"],  // Emotional tones present
            "visual_elements": ["string"],  // Notable visual elements
            "audio_elements": ["string"],  // Sound elements mentioned
            "genre": "string",  // Genre of the content
            "target_audience": ["string"],  // List of intended audiences
            "duration_estimate": "string",  // Estimated duration in minutes:seconds
            "quality_indicators": ["string"],  // Quality metrics or indicators
            "unique_identifiers": ["string"],  // Unique identifiers for the video
            "is_face_exist": bool,  // Whether faces are present in the video
            "person_identity": {{"name": "string", "gender": "string"}},  // Main person identity
            "other_person_identity": ["string"],  // Other persons' identities
            "psychological_personality": ["string"],  // Personality traits
            "no_of_person_in_video": int,  // Number of persons in the video 
            "content_warnings": ["string"],  // List of content warnings
            "safety_analysis": ["string"],  // Safety-related observations
            "is_safe": bool  // Whether the content is deemed safe
        }}
        
        Ensure all fields are filled based on the information available in the description.
        Return the response in valid JSON format.
        """
        
        # Make the API call
        """response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert content analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )"""

        #Changed as per GEMINI API

        # Initialize the Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Send the prompt to the Gemini API for content generation
        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(max_output_tokens=1000)
        )
        
        # Parse and return the response
        #print(response)
        #extracted_metadata = json.loads(response.text.strip())

        # Extract JSON text from the response
        raw_output = response.candidates[0].content.parts[0].text.strip()

        #Remove Markdown formatting (```json ... ```)
        cleaned_output = re.sub(r"^```json\n|\n```$", "", raw_output).strip()

        #Debugging: Print cleaned JSON before parsing
        print("Cleaned JSON Output:", cleaned_output)

        #Safely parse JSON
        extracted_metadata = json.loads(cleaned_output)


        if is_christian_content:
            christian_content = await analyze_christian_content(description, task_id)
            extracted_metadata["is_christian_content"] = christian_content

        logger.info("Extracted metadata:")
        logger.info(json.dumps(extracted_metadata, indent=2))
        return extracted_metadata

    except Exception as e:
        logger.error(f"Error during metadata extraction: {str(e)}")
        return {}
    

async def analyze_christian_content(description: str, task_id: str = None) -> dict:
    """
    Analyze a video description to determine the presence of Christian content.
    
    Args:
        description (str): Video description to analyze.
        task_id (str): Task identifier for progress tracking (optional).
        
    Returns:
        dict: Analysis results with the following fields:
            - is_christian: Boolean indicating if Christian content is present.
            - confidence_score: Float (0.0 to 1.0) representing confidence level.
            - indicators: List of specific Christian elements/themes detected.
    """
    try:
        # Define the prompt for GPT
        #system_prompt \
        prompt = f"""You are an expert in analyzing content for Christian themes.
        Based on the given description, identify if it contains Christian content.
        Provide the following details strictly in **JSON format**::
        1. "is_christian": Boolean value indicating if Christian content is present.
        2. "confidence_score": A numeric value (0.0-1.0) representing the certainty of Christian content presence.
        3. "indicators": A detailed list of specific elements/themes suggesting Christian content 
           (e.g., Bible verses, religious symbols, prayers, mentions of Jesus, etc.).

        Return the result strictly in the following JSON format:
        {
            "is_christian": boolean,
            "confidence_score": number,
            "indicators": [string]
        }
        If no Christian content is found, set: 
        {
            "is_christian": false,
            "confidence_score": 0.0,
            "indicators": []
        }
         
        Analyze the following description for Christian content:
        {description}
        """
        # user_message =

        # Call GPT model
        """response = await client.chat.completions.create(
            model="gpt-4",  # Fixed model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.3
        )"""

        #Changed as per GEMINI API

        # Initialize the Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Send the prompt to Gemini API for analysis
        response = model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(max_output_tokens=1000)
        )

        # Parse the response as JSON
        try:
            result = json.loads(response.text.strip())
            
            # Ensure "is_christian" is present
            if "is_christian" not in result:
                result["is_christian"] = False  # Default value if missing

            logger.info("Christian content analysis result:")
            logger.info(json.dumps(result, indent=2))
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            return {
                "is_christian": False,
                "confidence_score": 0.0,
                "indicators": []
            }

    except Exception as e:
        logger.error(f"Error in Christian content analysis: {str(e)}")
        return {
            "is_christian": False,
            "confidence_score": 0.0,
            "indicators": []
        }
    
