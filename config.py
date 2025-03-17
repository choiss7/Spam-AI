# API 키 및 설정 관리 파일
import os
from dotenv import load_dotenv

# .env 파일 로드 (있는 경우)
load_dotenv()

# OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Anthropic API 키
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Local AI 설정
LOCAL_AI_SETTINGS = {
    "base_url": "http://blue1.novamsg.org:4051/v1",
    "model": "/models/exaone",
    "api_key": os.getenv("LOCAL_AI_API_KEY", "")  # 필요한 경우 API 키 설정
}

# LLM 설정
LLM_SETTINGS = {
    "openai": {
        "model": "gpt-4o",  # 또는 다른 모델 (gpt-3.5-turbo, gpt-4 등)
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
    },
    "anthropic": {
        "model": "claude-3-opus-20240229",  # 또는 다른 모델 (claude-3-sonnet, claude-3-haiku 등)
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
    },
    "local_ai": {
        "model": LOCAL_AI_SETTINGS["model"],
        "max_tokens": 1024,
        "temperature": 0.0,  # 낮은 온도로 일관된 결과 유도
    }
}

# 스팸 분류 설정
SPAM_CLASSIFICATION_SETTINGS = {
    # 기본 샘플 크기 (None은 전체 데이터 사용)
    "default_sample_size": 10,
    
    # 기본 LLM 유형
    "default_llm_type": "local_ai",
    
    # API 호출 간 대기 시간 (초)
    "api_call_delay": 1,
    
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
    "spam_list": "스팸리스트_20250310_개인정보_삭제.xlsx"
} 