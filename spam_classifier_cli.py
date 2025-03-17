#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
스팸 분류 명령줄 인터페이스
LLM(GPT, Claude, Local AI)을 사용하여 스팸 메시지를 분류하는 CLI 도구
"""

import argparse
import os
import sys
from datetime import datetime

# 설정 파일 가져오기
from config import (
    SPAM_CLASSIFICATION_SETTINGS,
    LLM_SETTINGS,
    FILE_PATHS
)

# 스팸 분류 모듈 가져오기
try:
    from spam_classifier_llm import (
        classify_spam_messages,
        analyze_classification_results
    )
    # 사용 가능한 LLM 유형 확인
    from spam_classifier_llm import OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE
except ImportError as e:
    print(f"오류: 필요한 모듈을 가져올 수 없습니다: {e}")
    sys.exit(1)

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="LLM(GPT, Claude, Local AI)을 사용하여 스팸 메시지를 분류합니다."
    )
    
    # 필수 인수
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=FILE_PATHS["spam_list"],
        help=f"스팸 리스트 엑셀 파일 경로 (기본값: {FILE_PATHS['spam_list']})"
    )
    
    # 사용 가능한 LLM 유형 결정
    available_llm_types = ["local_ai"]
    if 'OPENAI_AVAILABLE' in globals() and OPENAI_AVAILABLE:
        available_llm_types.append("openai")
    if 'ANTHROPIC_AVAILABLE' in globals() and ANTHROPIC_AVAILABLE:
        available_llm_types.append("anthropic")
    
    # 선택적 인수
    parser.add_argument(
        "--llm", "-l",
        type=str,
        choices=available_llm_types,
        default="local_ai" if "local_ai" in available_llm_types else available_llm_types[0] if available_llm_types else None,
        help=f"사용할 LLM 유형 (기본값: local_ai)"
    )
    
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=SPAM_CLASSIFICATION_SETTINGS["default_sample_size"],
        help=f"분석할 샘플 크기 (기본값: {SPAM_CLASSIFICATION_SETTINGS['default_sample_size']}, 0은 전체 데이터 사용)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="결과를 저장할 폴더 경로 (기본값: 자동 생성)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력 모드 활성화"
    )
    
    return parser.parse_args()

def setup_output_folder(output_path=None):
    """결과를 저장할 폴더를 설정합니다."""
    if output_path is None:
        now = datetime.now()
        output_path = f"spam_llm_classification_{now.strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_path, exist_ok=True)
    return output_path

def display_settings(args, output_folder):
    """현재 설정을 표시합니다."""
    print("\n=== 스팸 분류 설정 ===")
    print(f"파일 경로: {args.file}")
    print(f"LLM 유형: {args.llm}")
    
    if args.llm == "local_ai":
        from config import LOCAL_AI_SETTINGS
        print(f"Local AI 기본 URL: {LOCAL_AI_SETTINGS['base_url']}")
        print(f"Local AI 모델: {LOCAL_AI_SETTINGS['model']}")
    else:
        print(f"LLM 모델: {LLM_SETTINGS[args.llm]['model']}")
    
    if args.sample == 0:
        print("샘플 크기: 전체 데이터")
    else:
        print(f"샘플 크기: {args.sample}")
    
    print(f"결과 저장 경로: {output_folder}")
    print("=====================\n")

def main():
    """메인 함수"""
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 샘플 크기 처리 (0은 전체 데이터 사용)
    sample_size = None if args.sample == 0 else args.sample
    
    # 결과 폴더 설정
    output_folder = setup_output_folder(args.output)
    
    # 설정 표시
    display_settings(args, output_folder)
    
    # 프롬프트 히스토리 파일 생성
    prompt_history_file = os.path.join(output_folder, "prompt_history.txt")
    with open(prompt_history_file, "w", encoding="utf-8") as f:
        f.write("# 스팸 리스트 LLM 분류 프롬프트 히스토리\n\n")
        f.write(f"## 분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"### 명령줄 인수:\n")
        f.write(f"- 파일 경로: {args.file}\n")
        f.write(f"- LLM 유형: {args.llm}\n")
        f.write(f"- 샘플 크기: {'전체 데이터' if sample_size is None else sample_size}\n")
        f.write(f"- 결과 저장 경로: {output_folder}\n\n")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {args.file}")
        
        # 스팸 분류 실행
        print(f"스팸 분류 시작...")
        df = classify_spam_messages(args.file, args.llm, sample_size)
        
        # 분류 결과 분석 및 시각화
        print(f"분류 결과 분석 중...")
        analyze_classification_results(df, output_folder)
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write(f"### 결과: LLM 스팸 분류 완료\n")
            f.write(f"- 분석된 메시지 수: {len(df)}\n")
            f.write(f"- 사용된 LLM: {args.llm}\n")
            
            if args.llm == "local_ai":
                from config import LOCAL_AI_SETTINGS
                f.write(f"- 사용된 모델: {LOCAL_AI_SETTINGS['model']}\n")
            else:
                f.write(f"- 사용된 모델: {LLM_SETTINGS[args.llm]['model']}\n")
                
            f.write(f"- 결과 저장 위치: {os.path.abspath(output_folder)}\n\n")
        
        print(f"\n분석이 완료되었습니다. 결과는 '{output_folder}' 폴더에 저장되었습니다.")
        
        # 결과 요약 표시
        if args.verbose:
            summary_file = os.path.join(output_folder, "classification_summary.txt")
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    print("\n=== 분류 결과 요약 ===")
                    print(f.read())
            else:
                print("\n요약 파일을 찾을 수 없습니다.")
    
    except FileNotFoundError as e:
        print(f"파일 오류: {e}")
        
        # 오류 정보 저장
        with open(os.path.join(output_folder, "error_log.txt"), "w", encoding="utf-8") as f:
            f.write(f"# 오류 정보\n\n")
            f.write(f"## 발생 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"## 오류 유형: 파일 오류\n")
            f.write(f"## 오류 메시지: {str(e)}\n")
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write(f"### 오류 발생: {str(e)}\n")
        
        return 1
    
    except Exception as e:
        print(f"오류 발생: {e}")
        
        # 오류 정보 저장
        with open(os.path.join(output_folder, "error_log.txt"), "w", encoding="utf-8") as f:
            f.write(f"# 오류 정보\n\n")
            f.write(f"## 발생 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"## 오류 메시지: {str(e)}\n")
            
            # 상세 오류 정보 추가
            import traceback
            f.write(f"\n## 상세 오류 정보:\n")
            f.write(f"```\n{traceback.format_exc()}\n```\n")
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write(f"### 오류 발생: {str(e)}\n")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 