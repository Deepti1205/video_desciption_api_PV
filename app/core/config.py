from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Video Description API"
    PROJECT_VERSION: str = "1.0.0"
    #OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    EMPOWERVERSE_API_KEY: str
    WEMOTIONS_API_KEY: str
    EMPOWERVERSE_API_PATH: str
    WEMOTIONS_API_PATH: str
    VIDEO_DESCRIPTION_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
#print(settings.dict())
#print(f"OPENAI_API_KEY loaded: {'*' * len(settings.OPENAI_API_KEY)}")