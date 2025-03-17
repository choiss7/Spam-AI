#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
스팸 메시지 분류 스크립트
LLM(GPT, Claude)을 사용하여 스팸 메시지를 분류합니다.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import tiktoken
import anthropic
from openai import OpenAI
import logging

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
    DEFAULT_LLM_PROVIDER,
    LLM_PRICING
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 한글 폰트 설정 (윈도우 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시스템 프롬프트 템플릿 업데이트 (config.py에서 가져온 것을 확장)
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE.replace(
    "\"explanation\": \"스팸 판단 이유에 대한 설명\"",
    "\"explanation\": \"스팸 판단 이유에 대한 설명\", \"spam_criteria\": \"스팸으로 분류한 주요 기준 (예: 피싱, 사기, 광고, 악성링크 등)\""
)

# 결과 폴더 생성
def create_results_folder(base_path: str = None) -> str:
    """결과를 저장할 폴더를 생성합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_path is None:
        base_path = FILE_PATHS.get("output_folder", "./spam_llm_classification")
    
    result_folder = f"{base_path}_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)
    return result_folder

# 프롬프트 히스토리 초기화
prompt_history = []

# OpenAI 토큰 카운터 초기화
def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """주어진 문자열의 토큰 수를 계산합니다."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception as e:
        logger.warning(f"토큰 계산 중 오류 발생: {e}")
        # 대략적인 토큰 수 추정 (영어 기준 1토큰 = 4자, 한글 기준 1토큰 = 1.5자 정도로 가정)
        return len(string) // 3

# 토큰 비용 계산 함수
def calculate_token_cost(token_usage: Dict[str, int], llm_type: str) -> Dict[str, float]:
    """토큰 사용량에 따른 비용을 계산합니다."""
    if llm_type not in LLM_PRICING:
        return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    
    pricing = LLM_PRICING[llm_type]
    input_cost = (token_usage.get("input_tokens", 0) / 1_000_000) * pricing["input"]
    output_cost = (token_usage.get("output_tokens", 0) / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# 데이터 로드 함수
def load_data(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    엑셀 파일에서 데이터를 로드하고 필요한 경우 샘플링합니다.
    
    Args:
        file_path (str): 엑셀 파일 경로
        sample_size (int, optional): 샘플 크기. None이면 전체 데이터 사용
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    try:
        # 엑셀 파일 로드
        df = pd.read_excel(file_path)
        logger.info(f"데이터 로드 완료: 총 {len(df)} 행")
        
        # 빈 메시지 제거
        valid_indices = []
        
        # 가능한 메시지 컬럼명들
        possible_message_columns = ["메시지내용", "message_content", "content", "내용"]
        message_column = None
        
        # 실제 존재하는 메시지 컬럼 찾기
        for col in possible_message_columns:
            if col in df.columns:
                message_column = col
                break
        
        if message_column is None:
            logger.error(f"메시지 컬럼을 찾을 수 없습니다. 가능한 컬럼명: {possible_message_columns}")
            # 샘플 데이터 추가
            sample_data = pd.DataFrame({
                "메시지내용": ["안녕하세요, 귀하의 계좌가 해킹되었습니다. 즉시 다음 링크를 클릭하여 정보를 업데이트하세요: http://fake-bank.com"],
                "스팸여부": ["스팸"]
            })
            return sample_data
        
        # 유효한 메시지만 필터링
        for idx, row in df.iterrows():
            message = row.get(message_column)
            if pd.isna(message) or message == "" or not isinstance(message, str):
                logger.warning(f"행 {idx}에 빈 메시지가 있습니다. 건너뜁니다.")
                continue
            
            valid_indices.append(idx)
        
        if valid_indices:
            valid_df = df.iloc[valid_indices].copy()
            logger.info(f"유효한 메시지 수: {len(valid_df)}")
        else:
            logger.warning("유효한 메시지가 없습니다. 샘플 데이터를 사용합니다.")
            # 샘플 데이터 추가
            valid_df = pd.DataFrame({
                message_column: ["안녕하세요, 귀하의 계좌가 해킹되었습니다. 즉시 다음 링크를 클릭하여 정보를 업데이트하세요: http://fake-bank.com"],
                "스팸여부": ["스팸"]
            })
        
        # 샘플링
        if sample_size is not None and sample_size > 0 and sample_size < len(valid_df):
            sampled_df = valid_df.sample(sample_size, random_state=42)
            logger.info(f"샘플링 완료: {len(sampled_df)} 행")
            return sampled_df
        
        return valid_df
        
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        # 오류 발생 시 샘플 데이터 반환
        sample_data = pd.DataFrame({
            "메시지내용": ["안녕하세요, 귀하의 계좌가 해킹되었습니다. 즉시 다음 링크를 클릭하여 정보를 업데이트하세요: http://fake-bank.com"],
            "스팸여부": ["스팸"]
        })
        return sample_data

# LLM 클라이언트 초기화
def init_llm_clients() -> Tuple[Optional[OpenAI], Optional[anthropic.Anthropic]]:
    """LLM API 클라이언트를 초기화합니다."""
    openai_client = None
    anthropic_client = None
    
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI 클라이언트가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 중 오류: {e}")
    
    if ANTHROPIC_API_KEY:
        try:
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("Anthropic 클라이언트가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"Anthropic 클라이언트 초기화 중 오류: {e}")
    
    return openai_client, anthropic_client

# 메시지 분류 함수 (OpenAI)
def classify_message_openai(client: OpenAI, message: str, system_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """OpenAI API를 사용하여 메시지를 분류합니다."""
    if not client:
        return {"error": "OpenAI 클라이언트가 초기화되지 않았습니다."}, {"input_tokens": 0, "output_tokens": 0, "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    
    try:
        # 시스템 프롬프트와 사용자 메시지 준비
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # 토큰 사용량 계산
        input_text = system_prompt + message
        input_tokens = num_tokens_from_string(input_text, LLM_SETTINGS["openai"]["model"])
        
        # API 호출
        start_time = time.time()
        response = client.chat.completions.create(
            model=LLM_SETTINGS["openai"]["model"],
            messages=messages,
            temperature=LLM_SETTINGS["openai"]["temperature"],
            max_tokens=LLM_SETTINGS["openai"]["max_tokens"],
            response_format={"type": "json_object"}
        )
        end_time = time.time()
        
        # 응답 처리
        response_text = response.choices[0].message.content
        output_tokens = response.usage.completion_tokens
        
        # 토큰 비용 계산
        token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        token_cost = calculate_token_cost(token_usage, "openai")
        
        # 프롬프트 히스토리에 추가
        prompt_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": LLM_SETTINGS["openai"]["model"],
            "provider": "openai",
            "input": message,
            "system_prompt": system_prompt,
            "output": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": token_cost["input_cost"],
            "output_cost": token_cost["output_cost"],
            "total_cost": token_cost["total_cost"],
            "response_time": end_time - start_time
        })
        
        # JSON 파싱
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 오류: {response_text}")
            result = {"error": "응답을 JSON으로 파싱할 수 없습니다.", "raw_response": response_text}
        
        # 토큰 사용량 및 비용 정보 추가
        usage_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": token_cost["input_cost"],
            "output_cost": token_cost["output_cost"],
            "total_cost": token_cost["total_cost"],
            "response_time": end_time - start_time
        }
        
        return result, usage_info
        
    except Exception as e:
        logger.error(f"OpenAI API 호출 중 오류: {str(e)}")
        return {"error": str(e)}, {"input_tokens": 0, "output_tokens": 0, "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

# 메시지 분류 함수 (Anthropic)
def classify_message_anthropic(client: anthropic.Anthropic, message: str, system_prompt: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Anthropic API를 사용하여 메시지를 분류합니다."""
    if not client:
        return {"error": "Anthropic 클라이언트가 초기화되지 않았습니다."}, {"input_tokens": 0, "output_tokens": 0, "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    
    try:
        # 토큰 사용량 계산
        input_text = system_prompt + message
        input_tokens = num_tokens_from_string(input_text)  # Anthropic 토큰 계산은 근사치
        
        # API 호출
        start_time = time.time()
        response = client.messages.create(
            model=LLM_SETTINGS["anthropic"]["model"],
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
            temperature=LLM_SETTINGS["anthropic"]["temperature"],
            max_tokens=LLM_SETTINGS["anthropic"]["max_tokens"],
            response_format={"type": "json_object"}
        )
        end_time = time.time()
        
        # 응답 처리
        response_text = response.content[0].text
        output_tokens = response.usage.output_tokens
        input_tokens = response.usage.input_tokens  # 실제 사용된 입력 토큰으로 업데이트
        
        # 토큰 비용 계산
        token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        token_cost = calculate_token_cost(token_usage, "anthropic")
        
        # 프롬프트 히스토리에 추가
        prompt_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": LLM_SETTINGS["anthropic"]["model"],
            "provider": "anthropic",
            "input": message,
            "system_prompt": system_prompt,
            "output": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": token_cost["input_cost"],
            "output_cost": token_cost["output_cost"],
            "total_cost": token_cost["total_cost"],
            "response_time": end_time - start_time
        })
        
        # JSON 파싱
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 오류: {response_text}")
            result = {"error": "응답을 JSON으로 파싱할 수 없습니다.", "raw_response": response_text}
        
        # 토큰 사용량 및 비용 정보 추가
        usage_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": token_cost["input_cost"],
            "output_cost": token_cost["output_cost"],
            "total_cost": token_cost["total_cost"],
            "response_time": end_time - start_time
        }
        
        return result, usage_info
            
    except Exception as e:
        logger.error(f"Anthropic API 호출 중 오류: {str(e)}")
        return {"error": str(e)}, {"input_tokens": 0, "output_tokens": 0, "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

# 스팸 분류 실행 함수
def run_spam_classification(
    file_path: str = None, 
    llm_type: str = "openai", 
    sample_size: int = None,
    output_folder: str = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """스팸 메시지 분류를 실행합니다."""
    # 파일 경로 설정
    if file_path is None:
        file_path = FILE_PATHS.get("spam_list_file", "./data/spam_list.xlsx")
    
    # 결과 폴더 생성
    result_folder = create_results_folder(output_folder)
    logger.info(f"결과 폴더가 생성되었습니다: {result_folder}")
    
    # LLM 클라이언트 초기화
    openai_client, anthropic_client = init_llm_clients()
    
    # 데이터 로드
    df = load_data(file_path, sample_size)
    
    # 시스템 프롬프트 설정
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    # 결과 저장용 리스트
    results = []
    total_token_usage = {"input_tokens": 0, "output_tokens": 0}
    
    # 각 메시지 분류
    for idx, row in df.iterrows():
        # 메시지 내용 추출
        message_content = ""
        for col in ["메시지내용", "message_content", "content", "내용"]:
            if col in row and not pd.isna(row[col]) and row[col].strip() != "":
                message_content = str(row[col])
                break
        
        # 빈 메시지 처리
        if pd.isna(message_content) or message_content.strip() == "":
            logger.warning(f"행 {idx}에 빈 메시지가 있습니다. 건너뜁니다.")
            continue
        
        # 진행 상황 출력
        if verbose:
            logger.info(f"메시지 {idx+1}/{len(df)} 분류 중...")
        else:
            if (idx + 1) % 10 == 0 or idx + 1 == len(df):
                logger.info(f"진행 상황: {idx+1}/{len(df)} ({((idx+1)/len(df))*100:.1f}%)")
        
        # LLM으로 분류
        result = {}
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        if llm_type == "openai" and openai_client:
            result, token_usage = classify_message_openai(openai_client, message_content, system_prompt)
        elif llm_type == "anthropic" and anthropic_client:
            result, token_usage = classify_message_anthropic(anthropic_client, message_content, system_prompt)
        else:
            logger.error(f"지원되지 않는 LLM 유형: {llm_type}")
            return {"error": f"지원되지 않는 LLM 유형: {llm_type}"}
        
        # 토큰 사용량 누적
        total_token_usage["input_tokens"] += token_usage["input_tokens"]
        total_token_usage["output_tokens"] += token_usage["output_tokens"]
        
        # 토큰 비용 계산
        token_cost = calculate_token_cost(token_usage, llm_type)
        
        # 결과 저장
        row_result = {
            "id": idx,
            "message": message_content,
            "classification": result,
            "spam_criteria": result.get("spam_criteria", ""),  # 스팸 분류 기준 추가
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
            "input_cost": token_cost["input_cost"],
            "output_cost": token_cost["output_cost"],
            "total_cost": token_cost["total_cost"]
        }
        
        # 원본 데이터의 필드 추가
        for col in df.columns:
            if col != "메시지내용" and col != "message_content" and col != "content" and col != "내용":
                row_result[col] = row[col]
        
        results.append(row_result)
        
        # API 호출 간 지연
        time.sleep(0.5)
    
    # 총 토큰 비용 계산
    total_cost = calculate_token_cost(total_token_usage, llm_type)
    
    # 결과 데이터프레임 생성
    results_df = pd.DataFrame(results)
    
    # 디버깅: 결과 데이터프레임의 컬럼 출력
    logger.info(f"결과 데이터프레임 컬럼: {list(results_df.columns)}")
    logger.info(f"결과 데이터프레임 행 수: {len(results_df)}")
    
    # 결과가 비어있는 경우 샘플 데이터 추가
    if len(results_df) == 0:
        logger.warning("결과 데이터프레임이 비어있습니다. 다시 메시지 처리를 시도합니다.")
        
        # 원본 데이터에서 처음 몇 개의 메시지 선택
        retry_count = min(5, len(df))
        for i in range(retry_count):
            row = df.iloc[i]
            
            # 메시지 내용 추출
            message_content = ""
            for col in ["메시지내용", "message_content", "content", "내용"]:
                if col in row and not pd.isna(row[col]) and row[col].strip() != "":
                    message_content = str(row[col])
                    break
            
            # 빈 메시지 건너뛰기
            if pd.isna(message_content) or message_content.strip() == "":
                continue
            
            logger.info(f"메시지 재처리 시도 {i+1}/{retry_count}: {message_content[:50]}...")
            
            # LLM으로 분류
            result = {}
            token_usage = {"input_tokens": 0, "output_tokens": 0}
            
            if llm_type == "openai" and openai_client:
                result, token_usage = classify_message_openai(openai_client, message_content, system_prompt)
            elif llm_type == "anthropic" and anthropic_client:
                result, token_usage = classify_message_anthropic(anthropic_client, message_content, system_prompt)
            
            # 토큰 사용량 누적
            total_token_usage["input_tokens"] += token_usage["input_tokens"]
            total_token_usage["output_tokens"] += token_usage["output_tokens"]
            
            # 토큰 비용 계산
            token_cost = calculate_token_cost(token_usage, llm_type)
            
            # 결과 저장
            row_result = {
                "id": row.name,
                "message": message_content,
                "classification": result,
                "spam_criteria": result.get("spam_criteria", ""),  # 스팸 분류 기준 추가
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "input_cost": token_cost["input_cost"],
                "output_cost": token_cost["output_cost"],
                "total_cost": token_cost["total_cost"]
            }
            
            # 원본 데이터의 필드 추가
            for col in df.columns:
                if col != "메시지내용" and col != "message_content" and col != "content" and col != "내용":
                    row_result[col] = row[col]
            
            results.append(row_result)
            
            # 성공적으로 처리된 경우 중단
            if isinstance(result, dict) and "is_spam" in result:
                break
            
            # API 호출 간 지연
            time.sleep(1.0)
        
        # 결과 데이터프레임 다시 생성
        if results:
            results_df = pd.DataFrame(results)
        else:
            # 여전히 결과가 없는 경우 샘플 데이터 추가
            logger.warning("재시도 후에도 결과가 없습니다. 샘플 데이터를 추가합니다.")
            sample_data = {
                "id": 0,
                "message": "샘플 메시지",
                "classification": {"is_spam": "비스팸", "spam_type": "샘플 데이터", "confidence": 0.0, "explanation": "샘플 데이터", "spam_criteria": ""},
                "is_spam": "비스팸",
                "spam_type": "샘플 데이터",
                "spam_criteria": "",  # 스팸 분류 기준 추가
                "confidence": 0.0,
                "explanation": "샘플 데이터",
                "input_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0
            }
            results_df = pd.DataFrame([sample_data])
    
    # 분류 결과 추출 및 통계 계산
    try:
        # 분류 결과 추출
        results_df["is_spam"] = results_df.apply(
            lambda row: row["classification"].get("is_spam", "알 수 없음") if isinstance(row.get("classification"), dict) else row.get("result", {}).get("is_spam", "알 수 없음") if isinstance(row.get("result"), dict) else "알 수 없음",
            axis=1
        )
        
        # spam_type 필드 추가 (category 필드를 사용하거나 없으면 기본값 사용)
        results_df["spam_type"] = results_df.apply(
            lambda row: row["classification"].get("spam_type", "알 수 없음") if isinstance(row.get("classification"), dict) else row.get("result", {}).get("category", "알 수 없음") if isinstance(row.get("result"), dict) else "알 수 없음",
            axis=1
        )
        
        # spam_criteria 필드 추가
        results_df["spam_criteria"] = results_df.apply(
            lambda row: row["classification"].get("spam_criteria", "") if isinstance(row.get("classification"), dict) else row.get("result", {}).get("spam_criteria", "") if isinstance(row.get("result"), dict) else "",
            axis=1
        )
        
        # spam_criteria가 비어있는 경우 explanation에서 가져오기
        results_df["spam_criteria"] = results_df.apply(
            lambda row: row["classification"].get("explanation", "") if (row["spam_criteria"] == "" or pd.isna(row["spam_criteria"])) and isinstance(row.get("classification"), dict)
                   else row.get("result", {}).get("reason", "") if (row["spam_criteria"] == "" or pd.isna(row["spam_criteria"])) and isinstance(row.get("result"), dict)
                   else row["spam_criteria"],
            axis=1
        )
        
        results_df["confidence"] = results_df.apply(
            lambda row: row["classification"].get("confidence", 0) if isinstance(row.get("classification"), dict) else row.get("result", {}).get("confidence", 0) if isinstance(row.get("result"), dict) else 0,
            axis=1
        )
        
        results_df["explanation"] = results_df.apply(
            lambda row: row["classification"].get("explanation", "") if isinstance(row.get("classification"), dict) else row.get("result", {}).get("reason", "") if isinstance(row.get("result"), dict) else "",
            axis=1
        )
        
        # 토큰 사용량 및 비용 추출
        results_df["input_tokens"] = results_df.apply(
            lambda row: row.get("token_usage", {}).get("input_tokens", 0) if isinstance(row.get("token_usage"), dict) else row.get("input_tokens", 0),
            axis=1
        )
        results_df["output_tokens"] = results_df.apply(
            lambda row: row.get("token_usage", {}).get("output_tokens", 0) if isinstance(row.get("token_usage"), dict) else row.get("output_tokens", 0),
            axis=1
        )
        results_df["input_cost"] = results_df.apply(
            lambda row: row.get("token_cost", {}).get("input_cost", 0.0) if isinstance(row.get("token_cost"), dict) else row.get("input_cost", 0.0),
            axis=1
        )
        results_df["output_cost"] = results_df.apply(
            lambda row: row.get("token_cost", {}).get("output_cost", 0.0) if isinstance(row.get("token_cost"), dict) else row.get("output_cost", 0.0),
            axis=1
        )
        results_df["total_cost"] = results_df.apply(
            lambda row: row.get("token_cost", {}).get("total_cost", 0.0) if isinstance(row.get("token_cost"), dict) else row.get("total_cost", 0.0),
            axis=1
        )
        
        # 스팸 통계 계산
        spam_count = results_df["is_spam"].apply(lambda x: "스팸" in str(x) if x is not None else False).sum()
        spam_ratio = spam_count / len(results_df) if len(results_df) > 0 else 0
        
        # 스팸 유형 분포
        if "spam_type" in results_df.columns:
            spam_types = results_df[results_df["is_spam"].apply(lambda x: "스팸" in str(x) if x is not None else False)]["spam_type"].value_counts()
        else:
            spam_types = pd.Series({"알 수 없음": spam_count})
        
        # 신뢰도 통계
        confidence_mean = results_df["confidence"].mean()
        confidence_std = results_df["confidence"].std()
        
        # 결과 저장
        results_df.to_csv(f"{result_folder}/classification_results.csv", index=False, encoding="utf-8-sig")
        
        # 프롬프트 히스토리 저장
        with open(f"{result_folder}/prompt_history.txt", "w", encoding="utf-8") as f:
            for prompt in prompt_history:
                f.write(f"시간: {prompt['timestamp']}\n")
                f.write(f"모델: {prompt['model']}\n")
                f.write(f"제공자: {prompt['provider']}\n")
                f.write(f"시스템 프롬프트: {prompt['system_prompt']}\n")
                f.write(f"입력 메시지: {prompt['input']}\n")
                f.write(f"출력 응답: {prompt['output']}\n")
                f.write(f"응답 시간: {prompt['response_time']:.2f}초\n")
                f.write(f"입력 토큰: {prompt['input_tokens']}\n")
                f.write(f"출력 토큰: {prompt['output_tokens']}\n")
                f.write(f"입력 토큰 비용: ${prompt['input_cost']:.6f}\n")
                f.write(f"출력 토큰 비용: ${prompt['output_cost']:.6f}\n")
                f.write(f"총 비용: ${prompt['total_cost']:.6f}\n")
                f.write("-" * 80 + "\n")
        
        # 분류 요약 저장
        with open(f"{result_folder}/classification_summary.txt", "w", encoding="utf-8") as f:
            f.write("스팸 분류 요약\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"분석 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터 파일: {file_path}\n")
            f.write(f"샘플 크기: {len(results_df)}\n")
            f.write(f"LLM 유형: {llm_type}\n")
            f.write(f"LLM 모델: {LLM_SETTINGS[llm_type]['model']}\n\n")
            
            f.write("스팸 통계:\n")
            f.write(f"- 스팸 메시지 수: {spam_count} ({spam_ratio:.1%})\n")
            f.write(f"- 비스팸 메시지 수: {len(results_df) - spam_count} ({1-spam_ratio:.1%})\n\n")
            
            f.write("스팸 유형 분포:\n")
            for spam_type, count in spam_types.items():
                f.write(f"- {spam_type}: {count} ({count/spam_count:.1%})\n")
            f.write("\n")
            
            # 스팸 분류 기준 분포 추가
            if "spam_criteria" in results_df.columns and spam_count > 0:
                spam_criteria_counts = results_df[results_df["is_spam"].apply(lambda x: "스팸" in str(x) if x is not None else False)]["spam_criteria"].value_counts()
                f.write("스팸 분류 기준 분포:\n")
                for criteria, count in spam_criteria_counts.items():
                    if criteria and criteria.strip():  # 빈 문자열이 아닌 경우에만 출력
                        f.write(f"- {criteria}: {count} ({count/spam_count:.1%})\n")
                f.write("\n")
            
            f.write("신뢰도 통계:\n")
            f.write(f"- 평균 신뢰도: {confidence_mean:.2f}\n")
            f.write(f"- 신뢰도 표준편차: {confidence_std:.2f}\n\n")
            
            f.write("토큰 사용량 및 비용:\n")
            f.write(f"- 총 입력 토큰: {total_token_usage['input_tokens']:,}\n")
            f.write(f"- 총 출력 토큰: {total_token_usage['output_tokens']:,}\n")
            f.write(f"- 총 토큰: {total_token_usage['input_tokens'] + total_token_usage['output_tokens']:,}\n")
            f.write(f"- 입력 토큰 비용: ${total_cost['input_cost']:.6f}\n")
            f.write(f"- 출력 토큰 비용: ${total_cost['output_cost']:.6f}\n")
            f.write(f"- 총 비용: ${total_cost['total_cost']:.6f}\n")
            f.write(f"- 메시지당 평균 비용: ${total_cost['total_cost']/len(results_df) if len(results_df) > 0 else 0:.6f}\n\n")
            
            f.write("비용 세부 정보:\n")
            f.write(f"- 사용 모델: {LLM_SETTINGS[llm_type]['model']}\n")
            f.write(f"- 제공자: {llm_type}\n")
            
            # 모델별 가격 정보
            if llm_type in LLM_PRICING:
                pricing = LLM_PRICING[llm_type]
                f.write(f"- 입력 토큰 가격: ${pricing['input']:.6f} / 백만 토큰\n")
                f.write(f"- 출력 토큰 가격: ${pricing['output']:.6f} / 백만 토큰\n")
            
            # 비용 통계
            if len(results_df) > 0:
                f.write(f"- 최소 메시지 비용: ${results_df['total_cost'].min():.6f}\n")
                f.write(f"- 최대 메시지 비용: ${results_df['total_cost'].max():.6f}\n")
                f.write(f"- 중간값 메시지 비용: ${results_df['total_cost'].median():.6f}\n")
                f.write(f"- 비용 표준편차: ${results_df['total_cost'].std():.6f}\n")
            
            f.write("\n참고: 비용은 USD 기준이며, 실제 비용은 API 제공업체의 가격 정책에 따라 다를 수 있습니다.\n")
            f.write("      ExaOne 모델은 무료로 설정되어 있습니다.\n")
        
        # 시각화: 카테고리 분포
        if len(spam_types) > 0:
            plt.figure(figsize=(10, 6))
            spam_types.plot(kind='bar')
            plt.title('스팸 유형 분포')
            plt.xlabel('스팸 유형')
            plt.ylabel('메시지 수')
            plt.tight_layout()
            plt.savefig(f"{result_folder}/category_distribution.png")
        else:
            logger.warning("스팸 유형 분포를 시각화할 데이터가 없습니다.")
            # 빈 차트 생성
            plt.figure(figsize=(10, 6))
            plt.title('스팸 유형 분포 (데이터 없음)')
            plt.xlabel('스팸 유형')
            plt.ylabel('메시지 수')
            plt.text(0.5, 0.5, '데이터가 없습니다', ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig(f"{result_folder}/category_distribution.png")
        
        # 시각화: 신뢰도 분포
        if len(results_df) > 0 and 'confidence' in results_df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(results_df['confidence'], bins=10, alpha=0.7)
            plt.axvline(confidence_mean, color='r', linestyle='--', label=f'평균: {confidence_mean:.2f}')
            plt.title('신뢰도 분포')
            plt.xlabel('신뢰도')
            plt.ylabel('메시지 수')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{result_folder}/confidence_distribution.png")
        else:
            logger.warning("신뢰도 분포를 시각화할 데이터가 없습니다.")
            # 빈 차트 생성
            plt.figure(figsize=(10, 6))
            plt.title('신뢰도 분포 (데이터 없음)')
            plt.xlabel('신뢰도')
            plt.ylabel('메시지 수')
            plt.text(0.5, 0.5, '데이터가 없습니다', ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig(f"{result_folder}/confidence_distribution.png")
            
        # 시각화: 비용 정보
        if len(results_df) > 0:
            plt.figure(figsize=(10, 6))
            
            # 비용 데이터 준비
            cost_data = {
                '입력 토큰 비용': total_cost['input_cost'],
                '출력 토큰 비용': total_cost['output_cost']
            }
            
            # 파이 차트 생성
            plt.pie(
                cost_data.values(), 
                labels=cost_data.keys(), 
                autopct='%1.1f%%',
                startangle=90,
                colors=['#66b3ff', '#ff9999']
            )
            plt.axis('equal')  # 원형 파이 차트를 위해
            plt.title(f'토큰 비용 분포 (총 ${total_cost["total_cost"]:.4f})')
            plt.tight_layout()
            plt.savefig(f"{result_folder}/token_cost_distribution.png")
            
            # 메시지별 비용 분포 히스토그램
            plt.figure(figsize=(10, 6))
            plt.hist(results_df['total_cost'], bins=10, alpha=0.7, color='#99ff99')
            plt.axvline(
                results_df['total_cost'].mean(), 
                color='r', 
                linestyle='--', 
                label=f'평균: ${results_df["total_cost"].mean():.6f}'
            )
            plt.title('메시지별 비용 분포')
            plt.xlabel('비용 (USD)')
            plt.ylabel('메시지 수')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{result_folder}/message_cost_distribution.png")
            
            # 토큰 사용량과 비용 관계 산점도
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df['input_tokens'] + results_df['output_tokens'],
                results_df['total_cost'],
                alpha=0.7,
                c='#ff9966'
            )
            plt.title('토큰 사용량과 비용 관계')
            plt.xlabel('총 토큰 수')
            plt.ylabel('비용 (USD)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{result_folder}/token_cost_relationship.png")
            
            # 스팸 분류 기준 분포 시각화 추가
            if "spam_criteria" in results_df.columns and spam_count > 0:
                spam_criteria_counts = results_df[results_df["is_spam"].apply(lambda x: "스팸" in str(x) if x is not None else False)]["spam_criteria"].value_counts()
                
                # 빈 문자열 제거
                spam_criteria_counts = spam_criteria_counts[spam_criteria_counts.index.str.strip() != ""]
                
                if len(spam_criteria_counts) > 0:
                    plt.figure(figsize=(12, 6))
                    spam_criteria_counts.plot(kind='bar')
                    plt.title('스팸 분류 기준 분포')
                    plt.xlabel('분류 기준')
                    plt.ylabel('메시지 수')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(f"{result_folder}/spam_criteria_distribution.png")
        else:
            logger.warning("비용 정보를 시각화할 데이터가 없습니다.")
        
        # 인간 분류와 LLM 분류 비교 (인간 분류 데이터가 있는 경우)
        if "human_classification" in results_df.columns:
            # 혼동 행렬 데이터 준비
            llm_vs_human = pd.crosstab(
                results_df["is_spam"].apply(lambda x: "스팸" in x), 
                results_df["human_classification"].apply(lambda x: "스팸" in str(x)),
                rownames=['LLM 분류'], 
                colnames=['인간 분류']
            )
            
            # 시각화: LLM vs 인간 분류
            plt.figure(figsize=(8, 6))
            plt.imshow(llm_vs_human, cmap='Blues')
            
            # 각 셀에 값 표시
            for i in range(llm_vs_human.shape[0]):
                for j in range(llm_vs_human.shape[1]):
                    plt.text(j, i, llm_vs_human.iloc[i, j], 
                            ha="center", va="center", color="black")
            
            plt.colorbar()
            plt.title('LLM vs 인간 분류 비교')
            plt.xticks([0, 1], ['비스팸', '스팸'])
            plt.yticks([0, 1], ['비스팸', '스팸'])
            plt.tight_layout()
            plt.savefig(f"{result_folder}/llm_vs_human_classification.png")
        
        # 결과 반환
        return {
            "status": "success",
            "result_folder": result_folder,
            "spam_count": int(spam_count),
            "spam_ratio": float(spam_ratio),
            "confidence_mean": float(confidence_mean)
        }
        
    except Exception as e:
        logger.error(f"결과 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"결과 처리 중 오류: {e}"}

# 메인 함수
if __name__ == "__main__":
    # 기본 파라미터
    file_path = FILE_PATHS.get("spam_list_file", "./data/spam_list.xlsx")
    llm_type = "openai"  # 기본값: OpenAI
    sample_size = None  # 전체 데이터 사용
    
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    if len(sys.argv) > 2:
        llm_type = sys.argv[2]
    if len(sys.argv) > 3:
        sample_size = int(sys.argv[3])
    
    # 스팸 분류 실행
    result = run_spam_classification(file_path, llm_type, sample_size)
    
    # 결과 출력
    if "error" in result:
        logger.error(f"오류: {result['error']}")
    else:
        logger.info(f"분류 완료! 결과 폴더: {result['result_folder']}")
        logger.info(f"스팸 비율: {result['spam_ratio']:.1%}")
        logger.info(f"평균 신뢰도: {result['confidence_mean']:.2f}") 