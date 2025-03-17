# API 키 및 설정 관리 파일
import os
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
load_dotenv()

# 기본 LLM 제공자 설정
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# OpenAI API 키 및 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MODEL_TOKEN_LIMIT = int(os.getenv("OPENAI_MODEL_TOKEN_LIMIT", "8192"))

# Anthropic API 키 및 설정
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
ANTHROPIC_MODEL_TOKEN_LIMIT = int(os.getenv("ANTHROPIC_MODEL_TOKEN_LIMIT", "100000"))

# Local AI 설정
LOCAL_AI_SETTINGS = {
    "base_url": os.getenv("LOCAL_AI_BASE_PATH", "http://localhost:8080/v1"),
    "model": os.getenv("LOCAL_AI_MODEL_PREF", "/models/exaone"),
    "api_key": os.getenv("LOCAL_AI_API_KEY", ""),
    "token_limit": int(os.getenv("LOCAL_AI_MODEL_TOKEN_LIMIT", "4096"))
}

# AWS Bedrock 설정 (Claude Sonnet 3.5 사용)
AWS_BEDROCK_SETTINGS = {
    "access_key_id": os.getenv("AWS_BEDROCK_LLM_ACCESS_KEY_ID", ""),
    "access_key": os.getenv("AWS_BEDROCK_LLM_ACCESS_KEY", ""),
    "region": os.getenv("AWS_BEDROCK_LLM_REGION", "ap-northeast-2"),
    "model": os.getenv("AWS_BEDROCK_LLM_MODEL_PREFERENCE", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    "token_limit": int(os.getenv("AWS_BEDROCK_LLM_MODEL_TOKEN_LIMIT", "4096"))
}

# ExaOne 설정
EXAONE_SETTINGS = {
    "base_url": os.getenv("EXAONE_BASE_PATH", LOCAL_AI_SETTINGS["base_url"]),
    "model": os.getenv("EXAONE_MODEL_PREF", "/models/exaone"),
    "api_key": os.getenv("EXAONE_API_KEY", LOCAL_AI_SETTINGS["api_key"]),
    "token_limit": int(os.getenv("EXAONE_MODEL_TOKEN_LIMIT", "4096"))
}

# LLM 설정
LLM_SETTINGS = {
    "openai": {
        "model": OPENAI_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
        "token_limit": OPENAI_MODEL_TOKEN_LIMIT
    },
    "anthropic": {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
        "token_limit": ANTHROPIC_MODEL_TOKEN_LIMIT
    },
    "claude-bedrock": {
        "model": AWS_BEDROCK_SETTINGS["model"],
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
        "token_limit": AWS_BEDROCK_SETTINGS["token_limit"]
    },
    "local_ai": {
        "model": LOCAL_AI_SETTINGS["model"],
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
        "token_limit": LOCAL_AI_SETTINGS["token_limit"]
    },
    "exaone": {
        "model": EXAONE_SETTINGS["model"],
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
        "token_limit": EXAONE_SETTINGS["token_limit"]
    }
}

# 스팸 분류 설정
SPAM_CLASSIFICATION_SETTINGS = {
    # 기본 샘플 크기 (None은 전체 데이터 사용)
    "default_sample_size": int(os.getenv("DEFAULT_SAMPLE_SIZE", "10")),
    
    # 기본 LLM 유형
    "default_llm_type": DEFAULT_LLM_PROVIDER,
    
    # API 호출 간 대기 시간 (초)
    "api_call_delay": float(os.getenv("API_CALL_DELAY", "1.0")),
    
    # 스팸 카테고리 목록
    "spam_categories": [
        "주식 및 투자",
        "결제 및 인증 코드",
        "온라인 서비스 광고",
        "교육",
        "오프라인 광고",
        "금융 상품",
        "기타"
    ]
}

# 시스템 프롬프트 템플릿
SYSTEM_PROMPT_TEMPLATE = """
당신은 스팸 메시지 분류 전문가입니다. 주어진 메시지를 분석하여 다음 카테고리 중 하나로 분류해주세요:
1. 스팸: 원치 않는 광고, 홍보, 피싱 등의 메시지
2. 비스팸: 정상적인 메시지

또한 스팸으로 분류한 경우, 다음 세부 카테고리 중 하나를 지정해주세요:
- 주식 및 투자: 주식, 부동산, 투자 관련 스팸
- 결제 및 인증 코드: 가짜 결제, 인증 관련 스팸
- 온라인 서비스 광고: 온라인 서비스 관련 광고 스팸
- 교육: 교육 관련 스팸
- 오프라인 광고: 오프라인 제품/서비스 광고 스팸
- 금융 상품: 대출, 보험 등 금융 상품 관련 스팸
- 기타: 위 카테고리에 속하지 않는 스팸

JSON 형식으로 다음 필드를 포함하여 응답해주세요:
{
    "is_spam": true/false,
    "category": "카테고리명",
    "confidence": 0.0~1.0 (확신도),
    "reason": "분류 이유에 대한 간단한 설명"
}
"""

# 파일 경로 설정
FILE_PATHS = {
    "spam_list": os.getenv("SPAM_LIST_PATH", "스팸리스트_20250310_개인정보_삭제.xlsx")
} 