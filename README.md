# 스팸 분류기 (LLM 기반)

이 프로젝트는 LLM(Large Language Model)을 활용하여 스팸 메시지를 자동으로 분류하는 도구입니다. OpenAI의 GPT, Anthropic의 Claude 모델, 그리고 Local AI(ExaOne)를 사용하여 스팸 메시지를 분석하고 카테고리별로 분류합니다.

## 주요 기능

- 엑셀 파일에서 스팸 메시지 데이터 읽기
- OpenAI GPT, Anthropic Claude 또는 Local AI를 사용한 스팸 분류
- 분류 결과 분석 및 시각화
- 기존 분류(AI, 휴먼)와 LLM 분류 비교
- 명령줄 인터페이스 제공

## 설치 방법

### 1. 필수 패키지 설치

설치 스크립트를 사용하여 필요한 패키지를 설치할 수 있습니다:

```bash
python install_requirements.py
```

또는 수동으로 설치할 수 있습니다:

```bash
pip install pandas openpyxl matplotlib seaborn python-dotenv requests
```

OpenAI 또는 Anthropic API를 사용하려면 추가 패키지가 필요합니다:

```bash
# OpenAI API 사용 시
pip install openai

# Anthropic API 사용 시
pip install anthropic
```

### 2. API 키 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음과 같이 API 키를 설정합니다:

```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
LOCAL_AI_API_KEY=your-local-ai-api-key  # 필요한 경우
```

## 사용 방법

### 명령줄 인터페이스 사용

```bash
python spam_classifier_cli.py --file "스팸리스트.xlsx" --llm local_ai --sample 10
```

#### 주요 옵션

- `--file`, `-f`: 스팸 리스트 엑셀 파일 경로 (기본값: config.py에 설정된 경로)
- `--llm`, `-l`: 사용할 LLM 유형 (`openai`, `anthropic` 또는 `local_ai`, 기본값: local_ai)
- `--sample`, `-s`: 분석할 샘플 크기 (기본값: config.py에 설정된 값, 0은 전체 데이터 사용)
- `--output`, `-o`: 결과를 저장할 폴더 경로 (기본값: 자동 생성)
- `--verbose`, `-v`: 상세 출력 모드 활성화

### Local AI 사용 예시

```bash
python spam_classifier_cli.py --file "스팸리스트.xlsx" --llm local_ai --sample 10
```

### 모듈로 사용

```python
from spam_classifier_llm import classify_spam_messages, analyze_classification_results

# 스팸 분류 실행 (Local AI 사용)
df = classify_spam_messages("스팸리스트.xlsx", "local_ai", 10)

# 분류 결과 분석 및 시각화
analyze_classification_results(df, "결과_폴더")
```

## 설정 파일 (config.py)

`config.py` 파일에서 다음 설정을 변경할 수 있습니다:

- API 키
- LLM 모델 및 파라미터
- Local AI 설정 (기본 URL, 모델 등)
- 스팸 분류 설정 (기본 샘플 크기, 기본 LLM 유형 등)
- 시스템 프롬프트 템플릿
- 파일 경로

### Local AI 설정 예시

```python
# Local AI 설정
LOCAL_AI_SETTINGS = {
    "base_url": "http://blue1.novamsg.org:4051/v1",
    "model": "/models/exaone",  # 모델 경로를 전체 경로로 지정
    "api_key": os.getenv("LOCAL_AI_API_KEY", "")  # 필요한 경우 API 키 설정
}
```

## 결과 파일

분석 결과는 다음 파일로 저장됩니다:

- `classification_results.csv`: 분류 결과가 포함된 CSV 파일
- `classification_summary.txt`: 분류 결과 요약
- `category_distribution.png`: 카테고리별 분포 시각화
- `confidence_distribution.png`: 확신도 분포 시각화
- `llm_vs_human_classification.png`: LLM 분류와 휴먼 분류 비교 (해당하는 경우)
- `prompt_history.txt`: 프롬프트 히스토리
- `error_log.txt`: 오류 발생 시 오류 정보

## 프로젝트 구조

- `spam_classifier_llm.py`: 스팸 분류 핵심 기능 구현
- `spam_classifier_cli.py`: 명령줄 인터페이스
- `config.py`: 설정 파일
- `install_requirements.py`: 필요한 패키지 설치 스크립트
- `README.md`: 프로젝트 설명

## 주의사항

- 이 도구는 OpenAI 및 Anthropic API를 사용하므로 API 사용량에 따라 비용이 발생할 수 있습니다.
- 대량의 데이터를 처리할 경우 API 호출 제한에 주의하세요.
- 개인정보가 포함된 데이터를 처리할 때는 관련 법규를 준수하세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 