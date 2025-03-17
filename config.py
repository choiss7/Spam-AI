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

# LLM 가격 정보 (USD 기준, 백만 토큰당)
LLM_PRICING = {
    "openai": {
        "input": float(os.getenv("OPENAI_PRICE_INPUT", "5.0")),  # GPT-4o 입력 토큰 가격: 백만 토큰당 $5
        "output": float(os.getenv("OPENAI_PRICE_OUTPUT", "15.0"))  # GPT-4o 출력 토큰 가격: 백만 토큰당 $15
    },
    "anthropic": {
        "input": float(os.getenv("ANTHROPIC_PRICE_INPUT", "15.0")),  # Claude 3 Opus 입력 토큰 가격: 백만 토큰당 $15
        "output": float(os.getenv("ANTHROPIC_PRICE_OUTPUT", "75.0"))  # Claude 3 Opus 출력 토큰 가격: 백만 토큰당 $75
    },
    "claude-bedrock": {
        "input": float(os.getenv("CLAUDE_BEDROCK_PRICE_INPUT", "3.0")),  # Claude 3.5 Sonnet 입력 토큰 가격: 백만 토큰당 $3
        "output": float(os.getenv("CLAUDE_BEDROCK_PRICE_OUTPUT", "15.0"))  # Claude 3.5 Sonnet 출력 토큰 가격: 백만 토큰당 $15
    },
    "local_ai": {
        "input": float(os.getenv("LOCAL_AI_PRICE_INPUT", "0.0")),  # Local AI 입력 토큰 가격: 무료
        "output": float(os.getenv("LOCAL_AI_PRICE_OUTPUT", "0.0"))  # Local AI 출력 토큰 가격: 무료
    },
    "exaone": {
        "input": float(os.getenv("EXAONE_PRICE_INPUT", "0.0")),  # ExaOne 입력 토큰 가격: 무료
        "output": float(os.getenv("EXAONE_PRICE_OUTPUT", "0.0"))  # ExaOne 출력 토큰 가격: 무료
    }
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
2. 스팸(표기법위반): [Web발신](광고) 표기법을 비준수한 메시지
3. 비스팸: 정상적인 메시지
4. 비스팸(표기법준수): 표기법을 준수한 광고 메시지

다음과 같은 예제는 표기법을 준수했기 때문에 비스팸으로 분류해주세요. (비스팸(표기법준수))
[Web발신](광고)스포힐상봉역점안녕하세요 회원님!스포힐휘트니스 상봉점(구 카인드라이프)을 이용해주셔서 진심으로 감사드립니다.3월 31일부터 시작된 '리모델링 오픈특가' 이벤트 안내 문자 보내드립니다.♥리모델링 오픈특가♥■ 1차 (3/10~4/6) 선착순 50명헬스 월 2만원! PT 4만원! ■ 2차 (4/7~4/27) 선착순 50명헬스 월 3만원! PT 4.5만원!■ 3차 (4/28~5/6) 선착순 50명헬스 월 4만원! PT 5만원!(vat 별도)※ 추가적으로 5월 7일 이후 부터는 회원권, PT 금액이 인상됨을 알려드립니다.※ 리모델링을 기다려주시는 회원님들께 보답드리고자 파격 할인을 진행중에 있으나 임차료, 인건비, 제반 비용 및 물가 상승으로 인해 지속적인 할인이 어려운점 회원님들의 넓은 이해와 양해 부탁드립니다.긴 글 읽어주셔서 감사합니다.무료거부 0808817402_x000D_ MMS/스포힐상봉역점

다음과 같은 예제는 표기법을 비준수했기 때문에 스팸으로 분류해주세요. (스팸(표기법위반))
(광고)최신상'셀루켓'액티브 50%사전예약 D-1lashevan.com무료수신거부0805301732_x000D_ SMS/-


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
    "is_spam": "스팸","스팸(표기법위반)","비스팸","비스팸(표기법준수)",
    "category": "카테고리명",
    "confidence": 0.0~1.0 (확신도),
    "reason": "분류 이유에 대한 간단한 설명"
}
"""

# 파일 경로 설정
FILE_PATHS = {
    "spam_list": os.getenv("SPAM_LIST_PATH", "./스팸리스트_20250310_개인정보_삭제.xlsx")
} 