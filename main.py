import os
import logging
import mimetypes
import base64
import httpx
import aiofiles
from aiofiles import os as aiofiles_os  # Import aiofiles.os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from telegram import Bot
from telegram.error import TelegramError, NetworkError

#### PROMPT ####
_prompt_ = '''
{
  "input": "Image captured by camera.",
  "task": "Analyze the image looking for birds.",
  "output_format": {
    "detected": "boolean",
    "description": "string (brief description if birds detected)"
  },
  "instructions": "Do you see little birds standing on the trees?",
  "example_output_1": {
    "detected": "yes",
    "description": "two little birds stands side by side on the tree."
  },
  "example_output_2": {
    "detected": "yes",
    "description": "a flying bird is flying away from the tree."
  },
  "example_output_3": {
    "detected": "no",
    "description": "no sign of birds on the trees."
  }
}
'''


################

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Telegram bot credentials
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise Exception("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID must be set as environment variables.")

# Initialize the bot
bot = Bot(token=TELEGRAM_TOKEN)

# GROQ API configurations
GROQ_API_URL = os.environ.get('GROQ_API_URL')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not GROQ_API_URL or not GROQ_API_KEY:
    raise Exception("GROQ_API_URL and GROQ_API_KEY must be set as environment variables.")

# Create FastAPI app
ROOT_PATH = os.getenv('ROOT_PATH', '/firebot')
app = FastAPI(root_path=ROOT_PATH)

# Directory to store uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed MIME types
ALLOWED_MIME_TYPES = {'image/jpeg'}

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    # Validate the uploaded file
    content_type = image.content_type
    if content_type not in ALLOWED_MIME_TYPES:
        logging.error("Invalid file format. Only JPEG images are accepted.")
        raise HTTPException(status_code=400, detail="Invalid file format, only JPEG images are accepted")

    # Secure the filename
    filename = os.path.basename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save the uploaded image to disk asynchronously
    async with aiofiles.open(filepath, "wb") as out_file:
        content = await image.read()  # async read
        await out_file.write(content)  # async write
    logging.info(f"Image saved at {filepath}")

    # Validate the file content (optional but recommended)
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type not in ALLOWED_MIME_TYPES:
        await aiofiles_os.remove(filepath)
        logging.error("Uploaded file content is not a valid JPEG image.")
        raise HTTPException(status_code=400, detail="Invalid image content")

    # Analyze the image using the GROQ API
#    prompt = "Please analyze this image."
    global _prompt_
    analysis_result = await analyze_image(filepath, _prompt_)

    if not analysis_result:
        await aiofiles_os.remove(filepath)
        logging.error("Failed to analyze the image.")
        raise HTTPException(status_code=500, detail="Failed to analyze the image")

    # Send the image and analysis result to Telegram
    try:
        await send_image_and_analysis_to_telegram(filepath, analysis_result)
    except Exception as e:
        logging.error(f"Failed to send image to Telegram: {e}")
        await aiofiles_os.remove(filepath)
        raise HTTPException(status_code=500, detail="Failed to send image to Telegram")

    # Remove the file after sending
    await aiofiles_os.remove(filepath)
    logging.info(f"Image file {filepath} removed after processing.")

    return JSONResponse(content={"message": "Image received, analyzed, and forwarded to Telegram"}, status_code=200)

async def analyze_image(image_path, prompt):
    try:
        # Read the image file asynchronously
        async with aiofiles.open(image_path, 'rb') as f:
            image_content = await f.read()

        # Encode the image
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Prepare the messages for Groq API
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]

            }
        ]

        # Make asynchronous request to Llama model
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GROQ_API_URL,
                json={
                    "model": "llama-3.2-11b-vision-preview",
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
            )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logging.info("Analysis result:")
            logging.info(answer)
            return answer
        else:
            logging.error(f"Error from Llama API: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logging.error(f"An unexpected error occurred in analyze_image: {str(e)}")
        return None

import datetime

async def send_image_and_analysis_to_telegram(image_path, analysis_result):
    try:
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Send the image with the timestamp as the caption
        with open(image_path, 'rb') as image_file:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=image_file,
                caption=timestamp
            )
        logging.info(f"Image sent to Telegram chat {TELEGRAM_CHAT_ID} with timestamp {timestamp}")

        # Send the analysis result as a separate message
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=analysis_result
        )
        logging.info(f"Analysis result sent to Telegram chat {TELEGRAM_CHAT_ID}")
    except (TelegramError, NetworkError) as e:
        logging.error(f"Error sending to Telegram: {e}")
        raise e
