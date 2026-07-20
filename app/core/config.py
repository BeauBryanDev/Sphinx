
from pathlib import Path
from tokenize import Single
from pydantic_settings import BaseSettings, SettingsConfigDict
 

# Single source of truth for runtime configuration. All paths, thresholds and
# toggles come from environment variables or a `.env` file at the repo root.
 
 
class Settings(BaseSettings):
        
    # Pipeline artifacts (paths relative to repo root by default)
        
    onnx_path        : Path = Path('artifacts/best_model_v9.onnx')
    class_map        : Path = Path('artifacts/class_map50_v9.json')
    trie_pkl         : Path = Path('artifacts/sphinx_trie_v4.pkl')
    bbaw_parquet     : Path = Path('artifacts/bbaw_clean.parquet')
    confusion_csv    : Path = Path('artifacts/confusion_matrix_v9_normalized.csv')
 
    
    # YOLO / ONNX inference parameters

    imgsz            : int   = 1024     # must match training imgsz
    conf_threshold   : float = 0.25     # min confidence to keep a detection
    iou_threshold    : float = 0.45     # NMS IoU threshold
    max_detections   : int   = 300      # cap per image after NMS
 

    # Pipeline parameters
    default_dir      : str   = 'rtl'    # 'rtl' or 'ltr'
    default_preset   : str   = 'default'
 

    # Request limits
 
    max_upload_mb    : int   = 20       # hard cap on uploaded image size
    request_timeout  : float = 30.0     # seconds before pipeline run aborts
 
 
    # CORS — comma-separated origins in env, parsed to list
 
    cors_origins     : list[str] = ['*']
 
    # OpenAI transliteration (Phase 6)
    # openai_api_key reads OPENAI_API_KEY from .env; empty = feature disabled
    openai_api_key      : str   = ''
    openai_model        : str   = 'gpt-4o'
    translit_max_signs  : int   = 20     # max Gardiner codes per LLM chunk
    translit_temperature: float = 0.1    # low — consistency over creativity

    # Metadata

    pipeline_version : str   = 'v9'
 
    model_config = SettingsConfigDict(
        env_file        = '.env',
        env_file_encoding = 'utf-8',
        case_sensitive  = False,
        extra           = 'ignore',
    )
 
 
# Single instance imported everywhere
settings = Settings()