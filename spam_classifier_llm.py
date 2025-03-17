import pandas as pd
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# 필요한 모듈 조건부 가져오기
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("경고: OpenAI 모듈을 가져올 수 없습니다. OpenAI 기능은 사용할 수 없습니다.")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("경고: Anthropic 모듈을 가져올 수 없습니다. Claude 기능은 사용할 수 없습니다.")

try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    print("경고: boto3 모듈을 가져올 수 없습니다. AWS Bedrock 기능은 사용할 수 없습니다.")

# 설정 파일 가져오기
from config import (
    OPENAI_API_KEY, 
    ANTHROPIC_API_KEY, 
    LOCAL_AI_SETTINGS,
    AWS_BEDROCK_SETTINGS,
    EXAONE_SETTINGS,
    LLM_SETTINGS, 
    SPAM_CLASSIFICATION_SETTINGS,
    SYSTEM_PROMPT_TEMPLATE,
    FILE_PATHS,
    DEFAULT_LLM_PROVIDER
)

# 한글 폰트 설정 (윈도우 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 결과 폴더 생성
now = datetime.now()
result_folder = f"spam_llm_classification_{now.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(result_folder, exist_ok=True)

# 프롬프트 히스토리 파일 생성 및 초기화
prompt_history_file = os.path.join(result_folder, "prompt_history.txt")
with open(prompt_history_file, "w", encoding="utf-8") as f:
    f.write("# 스팸 리스트 LLM 분류 프롬프트 히스토리\n\n")
    f.write(f"## 분석 시작 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("### 프롬프트: 스팸리스트 파일을 LLM(클로드, GPT)과 같은 NLP 기반의 분류를 하기 위한 소스 코드 작성\n\n")

# LLM 클라이언트 초기화 함수
def initialize_llm_clients():
    """LLM API 클라이언트를 초기화합니다."""
    clients = {}
    
    # OpenAI 클라이언트 초기화
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            clients["openai"] = OpenAI(api_key=OPENAI_API_KEY)
            print("OpenAI 클라이언트가 초기화되었습니다.")
        except Exception as e:
            print(f"OpenAI 클라이언트 초기화 실패: {e}")
    else:
        print("OpenAI API 키가 설정되지 않았거나 모듈이 설치되지 않았습니다.")
    
    # Anthropic 클라이언트 초기화
    if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
        try:
            clients["anthropic"] = Anthropic(api_key=ANTHROPIC_API_KEY)
            print("Anthropic 클라이언트가 초기화되었습니다.")
        except Exception as e:
            print(f"Anthropic 클라이언트 초기화 실패: {e}")
    else:
        print("Anthropic API 키가 설정되지 않았거나 모듈이 설치되지 않았습니다.")
    
    # AWS Bedrock 클라이언트 초기화 (Claude Sonnet 3.5 사용)
    if BEDROCK_AVAILABLE and AWS_BEDROCK_SETTINGS["access_key_id"] and AWS_BEDROCK_SETTINGS["access_key"]:
        try:
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=AWS_BEDROCK_SETTINGS["region"],
                aws_access_key_id=AWS_BEDROCK_SETTINGS["access_key_id"],
                aws_secret_access_key=AWS_BEDROCK_SETTINGS["access_key"]
            )
            clients["claude-bedrock"] = bedrock_runtime
            print("AWS Bedrock 클라이언트가 초기화되었습니다. (Claude Sonnet 3.5)")
        except Exception as e:
            print(f"AWS Bedrock 클라이언트 초기화 실패: {e}")
    else:
        print("AWS Bedrock 자격 증명이 설정되지 않았거나 모듈이 설치되지 않았습니다.")
    
    # Local AI 설정
    clients["local_ai"] = {
        "base_url": LOCAL_AI_SETTINGS["base_url"],
        "api_key": LOCAL_AI_SETTINGS["api_key"]
    }
    print("Local AI 설정이 초기화되었습니다.")
    
    # ExaOne 설정
    clients["exaone"] = {
        "base_url": EXAONE_SETTINGS["base_url"],
        "api_key": EXAONE_SETTINGS["api_key"]
    }
    print("ExaOne 설정이 초기화되었습니다.")
    
    return clients

# OpenAI GPT를 사용한 스팸 분류 함수
def classify_with_gpt(client, message_content, system_prompt=None):
    """
    OpenAI GPT를 사용하여 메시지를 스팸으로 분류합니다.
    
    Args:
        client: OpenAI 클라이언트
        message_content: 분류할 메시지 내용
        system_prompt: 시스템 프롬프트 (기본값: None)
        
    Returns:
        분류 결과 딕셔너리
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    try:
        # 설정에서 모델 및 파라미터 가져오기
        model = LLM_SETTINGS["openai"]["model"]
        max_tokens = LLM_SETTINGS["openai"]["max_tokens"]
        temperature = LLM_SETTINGS["openai"]["temperature"]
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # ```json 접두사 및 ``` 접미사 제거
        if "```json" in content:
            # ```json과 ``` 사이의 내용 추출
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```") and content.endswith("```"):
            # 일반 코드 블록 처리
            content = content[3:-3].strip()
        
        result = json.loads(content)
        result["model"] = model
        return result
    
    except Exception as e:
        print(f"GPT 분류 오류: {e}")
        return {
            "is_spam": None,
            "category": None,
            "confidence": 0,
            "reason": f"오류 발생: {str(e)}",
            "model": LLM_SETTINGS["openai"]["model"]
        }

# Anthropic Claude를 사용한 스팸 분류 함수
def classify_with_claude(client, message_content, system_prompt=None):
    """
    Anthropic Claude를 사용하여 메시지를 스팸으로 분류합니다.
    
    Args:
        client: Anthropic 클라이언트
        message_content: 분류할 메시지 내용
        system_prompt: 시스템 프롬프트 (기본값: None)
        
    Returns:
        분류 결과 딕셔너리
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    try:
        # 설정에서 모델 및 파라미터 가져오기
        model = LLM_SETTINGS["anthropic"]["model"]
        max_tokens = LLM_SETTINGS["anthropic"]["max_tokens"]
        temperature = LLM_SETTINGS["anthropic"]["temperature"]
        
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": message_content}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        content = response.content[0].text
        
        # ```json 접두사 및 ``` 접미사 제거
        if "```json" in content:
            # ```json과 ``` 사이의 내용 추출
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```") and content.endswith("```"):
            # 일반 코드 블록 처리
            content = content[3:-3].strip()
        
        result = json.loads(content)
        result["model"] = model
        return result
    
    except Exception as e:
        print(f"Claude 분류 오류: {e}")
        return {
            "is_spam": None,
            "category": None,
            "confidence": 0,
            "reason": f"오류 발생: {str(e)}",
            "model": LLM_SETTINGS["anthropic"]["model"]
        }

# Local AI를 사용한 스팸 분류 함수
def classify_with_local_ai(client_settings, message_content, system_prompt=None):
    """
    Local AI를 사용하여 메시지를 스팸으로 분류합니다.
    
    Args:
        client_settings: Local AI 설정
        message_content: 분류할 메시지 내용
        system_prompt: 시스템 프롬프트 (기본값: None)
        
    Returns:
        분류 결과 딕셔너리
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    try:
        base_url = client_settings["base_url"]
        api_key = client_settings["api_key"]
        model = LLM_SETTINGS["local_ai"]["model"]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            "temperature": LLM_SETTINGS["local_ai"]["temperature"],
            "max_tokens": LLM_SETTINGS["local_ai"]["max_tokens"]
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            try:
                # ```json 접두사 및 ``` 접미사 제거
                if "```json" in content:
                    # ```json과 ``` 사이의 내용 추출
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```") and content.endswith("```"):
                    # 일반 코드 블록 처리
                    content = content[3:-3].strip()
                
                result = json.loads(content)
                result["model"] = model
                return result
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {content}")
                return {
                    "is_spam": None,
                    "category": None,
                    "confidence": 0,
                    "reason": "JSON 파싱 오류",
                    "model": model
                }
        else:
            print(f"Local AI API 오류: {response.status_code} - {response.text}")
            return {
                "is_spam": None,
                "category": None,
                "confidence": 0,
                "reason": f"API 오류: {response.status_code}",
                "model": model
            }
    
    except Exception as e:
        print(f"Local AI 분류 오류: {e}")
        return {
            "is_spam": None,
            "category": None,
            "confidence": 0,
            "reason": f"오류 발생: {str(e)}",
            "model": LLM_SETTINGS["local_ai"]["model"]
        }

# ExaOne을 사용한 스팸 분류 함수
def classify_with_exaone(client_settings, message_content, system_prompt=None):
    """
    ExaOne을 사용하여 메시지를 스팸으로 분류합니다.
    
    Args:
        client_settings: ExaOne 클라이언트 설정
        message_content: 분류할 메시지 내용
        system_prompt: 시스템 프롬프트 (기본값: None)
        
    Returns:
        분류 결과 딕셔너리
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    try:
        base_url = client_settings["base_url"]
        api_key = client_settings["api_key"]
        model = LLM_SETTINGS["exaone"]["model"]
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content}
            ],
            "temperature": LLM_SETTINGS["exaone"]["temperature"],
            "max_tokens": LLM_SETTINGS["exaone"]["max_tokens"]
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            try:
                # ```json 접두사 및 ``` 접미사 제거
                if "```json" in content:
                    # ```json과 ``` 사이의 내용 추출
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```") and content.endswith("```"):
                    # 일반 코드 블록 처리
                    content = content[3:-3].strip()
                
                result = json.loads(content)
                result["model"] = model
                return result
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {content}")
                return {
                    "is_spam": None,
                    "category": None,
                    "confidence": 0,
                    "reason": "JSON 파싱 오류",
                    "model": model
                }
        else:
            print(f"ExaOne API 오류: {response.status_code} - {response.text}")
            return {
                "is_spam": None,
                "category": None,
                "confidence": 0,
                "reason": f"API 오류: {response.status_code}",
                "model": model
            }
    
    except Exception as e:
        print(f"ExaOne 분류 오류: {e}")
        return {
            "is_spam": None,
            "category": None,
            "confidence": 0,
            "reason": f"오류 발생: {str(e)}",
            "model": LLM_SETTINGS["exaone"]["model"]
        }

# AWS Bedrock을 통한 Claude Sonnet 3.5 사용 함수
def classify_with_claude_bedrock(client, message_content, system_prompt=None):
    """
    AWS Bedrock을 통해 Claude Sonnet 3.5를 사용하여 메시지를 스팸으로 분류합니다.
    
    Args:
        client: AWS Bedrock 클라이언트
        message_content: 분류할 메시지 내용
        system_prompt: 시스템 프롬프트 (기본값: None)
        
    Returns:
        분류 결과 딕셔너리
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    try:
        model_id = AWS_BEDROCK_SETTINGS["model"]
        
        # Claude 모델 형식
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": LLM_SETTINGS["claude-bedrock"]["max_tokens"],
            "temperature": LLM_SETTINGS["claude-bedrock"]["temperature"],
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": message_content}
            ]
        }
        
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response.get("body").read())
        content = response_body.get("content", [{}])[0].get("text", "")
        
        try:
            # ```json 접두사 및 ``` 접미사 제거
            if "```json" in content:
                # ```json과 ``` 사이의 내용 추출
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```") and content.endswith("```"):
                # 일반 코드 블록 처리
                content = content[3:-3].strip()
            
            result = json.loads(content)
            result["model"] = model_id
            return result
        except json.JSONDecodeError:
            print(f"JSON 파싱 오류: {content}")
            return {
                "is_spam": None,
                "category": None,
                "confidence": 0,
                "reason": "JSON 파싱 오류",
                "model": model_id
            }
    
    except Exception as e:
        print(f"Claude Bedrock 분류 오류: {e}")
        return {
            "is_spam": None,
            "category": None,
            "confidence": 0,
            "reason": f"오류 발생: {str(e)}",
            "model": AWS_BEDROCK_SETTINGS["model"]
        }

# 스팸 분류 실행 함수
def classify_spam_messages(file_path, llm_type=None, sample_size=None):
    """
    엑셀 파일에서 메시지를 읽고 LLM을 사용하여 스팸을 분류합니다.
    
    Args:
        file_path: 스팸 리스트 엑셀 파일 경로
        llm_type: 사용할 LLM 유형 ('openai', 'anthropic', 'claude-bedrock', 'local_ai', 'exaone')
        sample_size: 샘플 크기 (None인 경우 전체 데이터 사용)
        
    Returns:
        분류 결과가 추가된 데이터프레임
    """
    # 기본값 설정
    if llm_type is None:
        llm_type = SPAM_CLASSIFICATION_SETTINGS["default_llm_type"]
    
    if sample_size is None:
        sample_size = SPAM_CLASSIFICATION_SETTINGS["default_sample_size"]
    
    print(f"파일 '{file_path}' 읽는 중...")
    
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path)
    print(f"데이터 크기: {df.shape}")
    
    # 샘플링 (필요한 경우)
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"샘플 크기: {df.shape}")
    
    # LLM 클라이언트 초기화
    clients = initialize_llm_clients()
    
    if llm_type not in clients:
        raise ValueError(f"지원되지 않는 LLM 유형 또는 API 키가 설정되지 않음: {llm_type}")
    
    # 결과 저장을 위한 열 추가
    df["llm_is_spam"] = None
    df["llm_category"] = None
    df["llm_confidence"] = None
    
    # llm_reason을 모델별 필드로 변경
    if llm_type == "openai":
        df["gpt_reason"] = None
    elif llm_type == "anthropic":
        df["claude_reason"] = None
        df["exaone_reason"] = None
    elif llm_type == "claude-bedrock":
        df["claude_reason"] = None
    elif llm_type == "exaone":
        df["exaone_reason"] = None
    else:  # local_ai
        df["local_ai_reason"] = None
        
    df["llm_model"] = None
    
    # 분류 함수 선택
    if llm_type == "openai":
        classify_func = lambda msg: classify_with_gpt(clients["openai"], msg)
    elif llm_type == "anthropic":
        classify_func = lambda msg: classify_with_claude(clients["anthropic"], msg)
    elif llm_type == "claude-bedrock":
        classify_func = lambda msg: classify_with_claude_bedrock(clients["claude-bedrock"], msg)
    elif llm_type == "exaone":
        classify_func = lambda msg: classify_with_exaone(clients["exaone"], msg)
    else:  # local_ai
        classify_func = lambda msg: classify_with_local_ai(clients["local_ai"], msg)
    
    # 각 메시지 분류
    total = len(df)
    for idx, (i, row) in enumerate(df.iterrows()):
        print(f"메시지 분류 중... {idx+1}/{total}")
        
        # 메시지 내용 추출
        try:
            message = str(row["내용"]) if pd.notna(row["내용"]) else ""
            if not message:
                print(f"경고: 메시지 {idx+1}의 내용이 비어 있습니다.")
                df.at[i, "llm_is_spam"] = "비스팸"
                df.at[i, "llm_category"] = "분류 불가"
                df.at[i, "llm_confidence"] = 0.0
                
                # 모델별 reason 필드 설정
                if llm_type == "openai":
                    df.at[i, "gpt_reason"] = "메시지 내용이 비어 있습니다."
                elif llm_type == "anthropic":
                    df.at[i, "claude_reason"] = "메시지 내용이 비어 있습니다."
                    df.at[i, "exaone_reason"] = "메시지 내용이 비어 있습니다."
                elif llm_type == "claude-bedrock":
                    df.at[i, "claude_reason"] = "메시지 내용이 비어 있습니다."
                elif llm_type == "exaone":
                    df.at[i, "exaone_reason"] = "메시지 내용이 비어 있습니다."
                else:
                    df.at[i, "local_ai_reason"] = "메시지 내용이 비어 있습니다."
                    
                df.at[i, "llm_model"] = LLM_SETTINGS[llm_type]["model"]
                continue
        except KeyError:
            print(f"경고: '내용' 열을 찾을 수 없습니다. 데이터프레임 열: {df.columns.tolist()}")
            raise ValueError("데이터프레임에 '내용' 열이 없습니다. 파일 형식을 확인하세요.")
        
        # LLM으로 분류
        result = classify_func(message)
        
        # 결과 저장
        df.at[i, "llm_is_spam"] = "스팸" if result.get("is_spam") else "비스팸"
        df.at[i, "llm_category"] = result.get("category")
        df.at[i, "llm_confidence"] = result.get("confidence")
        
        # 모델별 reason 필드 설정
        if llm_type == "openai":
            df.at[i, "gpt_reason"] = result.get("reason")
        elif llm_type == "anthropic":
            df.at[i, "claude_reason"] = result.get("reason")
            df.at[i, "exaone_reason"] = result.get("reason")
        elif llm_type == "claude-bedrock":
            df.at[i, "claude_reason"] = result.get("reason")
        elif llm_type == "exaone":
            df.at[i, "exaone_reason"] = result.get("reason")
        else:
            df.at[i, "local_ai_reason"] = result.get("reason")
            
        df.at[i, "llm_model"] = result.get("model")
        
        # API 호출 제한을 위한 대기 시간
        time.sleep(SPAM_CLASSIFICATION_SETTINGS["api_call_delay"])
    
    return df

# 분류 결과 분석 및 시각화 함수
def analyze_classification_results(df, result_folder, llm_type=None):
    """
    분류 결과를 분석하고 시각화합니다.
    
    Args:
        df: 분류 결과가 포함된 데이터프레임
        result_folder: 결과를 저장할 폴더 경로
        llm_type: 사용된 LLM 유형 (openai, anthropic, claude-bedrock, local_ai, exaone)
    """
    # 결과 저장 전에 모델별 reason 필드 처리
    # 모델별 reason 필드가 있는지 확인하고 없으면 빈 열 추가
    if llm_type == "openai" and "gpt_reason" not in df.columns:
        df["gpt_reason"] = None
    elif llm_type == "anthropic":
        if "claude_reason" not in df.columns:
            df["claude_reason"] = None
        if "exaone_reason" not in df.columns:
            df["exaone_reason"] = None
    elif llm_type == "claude-bedrock" and "claude_reason" not in df.columns:
        df["claude_reason"] = None
    elif llm_type == "exaone" and "exaone_reason" not in df.columns:
        df["exaone_reason"] = None
    elif llm_type == "local_ai" and "local_ai_reason" not in df.columns:
        df["local_ai_reason"] = None
    
    # 결과 저장
    df.to_csv(os.path.join(result_folder, "classification_results.csv"), index=False, encoding="utf-8-sig")
    
    # 기본 통계 계산
    # "스팸"/"비스팸" 문자열 기반으로 계산
    spam_count = (df["llm_is_spam"] == "스팸").sum()
    total_count = len(df)
    spam_ratio = spam_count / total_count if total_count > 0 else 0
    
    # 카테고리별 분포 (None 값 제외)
    df["llm_category_clean"] = df["llm_category"].fillna("분류 없음")
    category_counts = df["llm_category_clean"].value_counts()
    
    # LLM 분류와 기존 분류(있는 경우) 비교
    comparison_df = None
    if "AI 분류" in df.columns and "휴먼 분류" in df.columns:
        comparison_df = pd.DataFrame({
            "LLM 분류": df["llm_category_clean"],
            "AI 분류": df["AI 분류"].fillna("분류 없음"),
            "휴먼 분류": df["휴먼 분류"].fillna("분류 없음")
        })
    
    # 결과 요약 저장
    with open(os.path.join(result_folder, "classification_summary.txt"), "w", encoding="utf-8") as f:
        f.write("# LLM 스팸 분류 결과 요약\n\n")
        f.write(f"## 기본 통계\n")
        f.write(f"- 총 메시지 수: {total_count}\n")
        f.write(f"- 스팸으로 분류된 메시지 수: {spam_count}\n")
        f.write(f"- 스팸 비율: {spam_ratio:.2%}\n\n")
        
        f.write(f"## 카테고리별 분포\n")
        for category, count in category_counts.items():
            f.write(f"- {category}: {count} ({count/total_count:.2%})\n")
        
        if comparison_df is not None:
            f.write(f"\n## LLM 분류와 기존 분류 비교\n")
            f.write("LLM 분류와 휴먼 분류 일치율: ")
            match_rate = (comparison_df["LLM 분류"] == comparison_df["휴먼 분류"]).mean()
            f.write(f"{match_rate:.2%}\n")
    
    # 시각화: 카테고리별 분포
    if len(category_counts) > 0:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title('LLM 분류 카테고리 분포')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, "category_distribution.png"))
        plt.close()
    else:
        print("경고: 카테고리 분포를 시각화할 데이터가 없습니다.")
    
    # 시각화: 확신도 분포 (None 값 제외)
    df["llm_confidence_clean"] = df["llm_confidence"].apply(lambda x: float(x) if x is not None else 0.0)
    if df["llm_confidence_clean"].count() > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df["llm_confidence_clean"], bins=20)
        plt.title('LLM 분류 확신도 분포')
        plt.xlabel('확신도')
        plt.ylabel('빈도')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, "confidence_distribution.png"))
        plt.close()
    else:
        print("경고: 확신도 분포를 시각화할 데이터가 없습니다.")
    
    # 시각화: LLM vs 휴먼 분류 (있는 경우)
    if comparison_df is not None and len(comparison_df) > 0:
        try:
            plt.figure(figsize=(12, 8))
            confusion_matrix = pd.crosstab(
                comparison_df["LLM 분류"], 
                comparison_df["휴먼 분류"],
                normalize="index"
            )
            sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".2f")
            plt.title('LLM 분류 vs 휴먼 분류')
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, "llm_vs_human_classification.png"))
            plt.close()
        except Exception as e:
            print(f"휴먼 분류 비교 시각화 오류: {e}")
            # 오류 정보 저장
            with open(os.path.join(result_folder, "visualization_error.txt"), "w", encoding="utf-8") as f:
                f.write(f"# 시각화 오류 정보\n\n")
                f.write(f"## 오류 메시지: {str(e)}\n")

# 메인 함수
def main():
    """메인 함수"""
    try:
        # 파일 경로 설정
        file_path = FILE_PATHS["spam_list"]
        
        # 사용자 입력 받기 (실제 사용 시 구현)
        print("스팸 분류 설정:")
        print(f"1. 기본 LLM 유형: {SPAM_CLASSIFICATION_SETTINGS['default_llm_type']}")
        print(f"2. 기본 샘플 크기: {SPAM_CLASSIFICATION_SETTINGS['default_sample_size']}")
        
        # 여기서는 기본값 사용
        llm_type = SPAM_CLASSIFICATION_SETTINGS["default_llm_type"]
        sample_size = SPAM_CLASSIFICATION_SETTINGS["default_sample_size"]
        
        # 스팸 분류 실행
        df = classify_spam_messages(file_path, llm_type, sample_size)
        
        # 분류 결과 분석 및 시각화
        analyze_classification_results(df, result_folder, llm_type)
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write(f"### 결과: LLM 스팸 분류 완료\n")
            f.write(f"- 분석된 메시지 수: {len(df)}\n")
            f.write(f"- 사용된 LLM: {llm_type}\n")
            f.write(f"- 사용된 모델: {LLM_SETTINGS[llm_type]['model']}\n")
            f.write(f"- 결과 저장 위치: {os.path.abspath(result_folder)}\n\n")
        
        print(f"분석이 완료되었습니다. 결과는 '{result_folder}' 폴더에 저장되었습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        
        # 오류 정보 저장
        with open(os.path.join(result_folder, "error_log.txt"), "w", encoding="utf-8") as f:
            f.write(f"# 오류 정보\n\n")
            f.write(f"## 발생 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"## 오류 메시지: {str(e)}\n")
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write(f"### 오류 발생: {str(e)}\n")

# 스크립트가 직접 실행될 때만 메인 함수 호출
if __name__ == "__main__":
    main() 