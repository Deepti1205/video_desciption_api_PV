# 🎥 Video Description API

![Video Description API](https://img.shields.io/badge/API-Video%20Description-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project provides a powerful API for analyzing videos, generating detailed descriptions, and extracting relevant keywords. It processes both video and audio content in parallel for efficient and comprehensive analysis.

## 📋 Table of Contents

1. [Features](#-features)
2. [Installation](#-installation)
3. [Usage](#-usage)
4. [API Endpoints](#-api-endpoints)
5. [Examples](#-examples)
   - [Python](#python)
   - [JavaScript](#javascript)
   - [cURL](#curl)
6. [Configuration](#-configuration)
7. [Contributing](#-contributing)
8. [License](#-license)

## 🌟 Features

- Parallel processing of video and audio content
- Extraction of key frames from videos
- Audio transcription using OpenAI's Whisper API
- Detailed video description generation using GPT-4
- Keyword extraction with relevance weighting
- Asynchronous processing with task ID for result retrieval
- Support for both file uploads and URL-based video processing

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/video-description-api.git
   cd video-description-api
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here # Please use your GEMINI API key
   EMPOWERVERSE_API_KEY=your_empowerverse_api_key_here
   API_PATH=your_api_path_here
   ```

5. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`.

## 🎬 Usage

The API provides several endpoints for video analysis and result retrieval. Here's a brief overview of each endpoint:

## 🛠 API Endpoints

### POST /api/v1/analyze_video

Upload a video file or provide a URL to start the analysis process.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - `video`: The video file to analyze (optional)
  - `file_url`: URL of the video to analyze (optional)
  - `identifier`: Custom identifier for the video (optional)

**Response:**
```json
{
  "message": "Video analysis started.",
  "task_id": "unique_task_id"
}
```

### GET /api/v1/analysis_result/{task_id}

Retrieve the analysis results for a given task ID.

**Request:**
- Method: GET
- Path Parameter: 
  - `task_id`: The unique task ID returned from the analyze_video endpoint

**Response:**
```json
{
  "status": "completed",
  "description": "Detailed description of the video content",
  "keywords": [
    {
      "keyword": "example keyword",
      "weight": 10
    },
    ...
  ],
  "topics": ["topic1", "topic2", ...],
  "entities": ["entity1", "entity2", ...],
  "actions": ["action1", "action2", ...],
  "emotions": ["emotion1", "emotion2", ...],
  "visual_elements": ["element1", "element2", ...],
  "audio_elements": ["element1", "element2", ...],
  "genre": "video genre",
  "target_audience": ["audience1", "audience2", ...],
  "duration_estimate": "estimated duration",
  "quality_indicators": ["indicator1", "indicator2", ...],
  "unique_identifiers": ["identifier1", "identifier2", ...],
  "is_face_exist": true/false,
  "person_identity": "person name and gender",
  "other_person_identity": ["person1 name and gender", "person2 name and gender", ...],
  "psychological_personality": ["trait1", "trait2", "trait3"],
  "no_of_person_in_video": number,
  "transcript": "full transcript with timestamps"
}
```

### POST /api/v1/share_url

Process a video from a given URL.

**Request:**
- Method: POST
- Headers:
  - `flic_token`: Authentication token
- Body:
  ```json
  {
    "url": "https://example.com/video.mp4",
    "identifier": "custom_identifier"
  }
  ```

**Response:**
```json
{
  "status": "success",
  "message": "URL processed successfully, video processing in queue..."
}
```

## 💻 Examples

### Python

```python
import requests
import json
import time
import os

MAX_RETRIES = 20
RETRY_DELAY = 5

def start_analysis(video_path):
    url = "http://13.92.184.232:8000/api/v1/analyze_video"

    if video_path.startswith(('http://', 'https://')):
        params = {"file_url": video_path}
        response = requests.post(url, params=params)
    else:
        with open(video_path, "rb") as video_file:
            files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
            response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
        return response.json().get('task_id')
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def get_analysis_result(task_id):
    url = f"http://13.92.184.232:8000/api/v1/analysis_result/{task_id}"
    for attempt in range(MAX_RETRIES):
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'completed':
                return result
            elif result.get('status') == 'error':
                print(f"Error in analysis: {result.get('message')}")
                return None
        print(f"Attempt {attempt + 1}: Result not ready. Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
    print("Max retries reached. Analysis result not available.")
    return None

def main():
    video_path = "https://video-cdn.socialverseapp.com/swapnil_e1994ef0-93e5-4c2e-9f53-1c7d63e1bb82.mp4"
    task_id = start_analysis(video_path)
    if task_id:
        result = get_analysis_result(task_id)
        if result:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### JavaScript

```javascript
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const MAX_RETRIES = 20;
const RETRY_DELAY = 5000; // 5 seconds in milliseconds

async function startAnalysis(videoPath) {
    const url = 'http://13.92.184.232:8000/api/v1/analyze_video';
    let data;
    let headers = {};

    if (videoPath.startsWith('http://') || videoPath.startsWith('https://')) {
        data = { file_url: videoPath };
    } else {
        const formData = new FormData();
        formData.append('video', fs.createReadStream(videoPath), {
            filename: videoPath,
            contentType: 'video/mp4',
        });
        data = formData;
        headers = formData.getHeaders();
    }

    try {
        const response = await axios.post(url, data, { headers });
        console.log(JSON.stringify(response.data, null, 2));
        return response.data.task_id;
    } catch (error) {
        console.error('Error starting analysis:', error.message);
        return null;
    }
}

async function getAnalysisResult(taskId) {
    const url = `http://13.92.184.232:8000/api/v1/analysis_result/${taskId}`;
    
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await axios.get(url);
            const result = response.data;
            
            if (result.status === 'completed') {
                return result;
            } else if (result.status === 'error') {
                console.error('Error in analysis:', result.message);
                return null;
            }
            
            console.log(`Attempt ${attempt + 1}: Result not ready. Retrying in ${RETRY_DELAY / 1000} seconds...`);
            await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        } catch (error) {
            console.error('Error getting analysis result:', error.message);
            return null;
        }
    }
    
    console.log('Max retries reached. Analysis result not available.');
    return null;
}

async function main() {
    const videoPath = 'https://video-cdn.socialverseapp.com/swapnil_433eadbc-64a6-4fd5-84aa-1c33ba85971c.mp4';
    const taskId = await startAnalysis(videoPath);
    if (taskId) {
        const result = await getAnalysisResult(taskId);
        if (result) {
            console.log(JSON.stringify(result, null, 2));
        }
    }
}

main();
```

### cURL

You can use cURL to interact with the API directly from the command line:

1. Uploading a video and starting analysis:

```bash
#curl -X POST -F "video=@path/to/your/video.mp4" http://13.92.184.232:8000/api/v1/analyze_video

Please use this to test
curl -X POST -F "video=@path/to/your/swapnil_e1994ef0-93e5-4c2e-9f53-1c7d63e1bb82.mp4" http://127.0.0.1:8000/api/v1/analyze_video?app_name=app
```

2. Retrieving analysis results:

```bash
curl http://13.92.184.232:8000/api/v1/analysis_result/your_task_id_here
```

Replace `your_task_id_here` with the actual `task_id` received from the previous command.

3. Processing a video from a URL:

```bash
curl -X POST -H "Content-Type: application/json" -H "flic-token: your_flic_token_here" -d '{"url": "https://example.com/video.mp4", "identifier": "custom_identifier"}' http://13.92.184.232:8000/api/v1/share_url

Please use this to test
curl -X POST -H "Content-Type:application/json" -d "{\"url\":\"https://video-cdn.socialverseapp.com/swapnil_e1994ef0-93e5-4c2e-9f53-1c7d63e1bb82.mp4\", \"identifier\":\"1\"}" http://127.0.0.1:8000/api/v1/share_url?app_name=app
```

Replace `your_flic_token_here` with your actual authentication token.

## ⚙️ Configuration

The project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
EMPOWERVERSE_API_KEY=your_empowerverse_api_key_here
API_PATH=your_api_path_here
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
