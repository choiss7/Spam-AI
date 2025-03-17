#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
스팸 분류 명령줄 인터페이스
LLM(GPT, Claude, Claude Sonnet 3.5, ExaOne, Local AI)을 사용하여 스팸 메시지를 분류하는 CLI 도구
"""

import argparse
import os
import sys
import subprocess
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
    from spam_classifier_llm import OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE, BEDROCK_AVAILABLE
except ImportError as e:
    print(f"오류: 필요한 모듈을 가져올 수 없습니다: {e}")
    sys.exit(1)

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="LLM(GPT, Claude, Claude Sonnet 3.5, ExaOne, Local AI)을 사용하여 스팸 메시지를 분류합니다."
    )
    
    # 필수 인수
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=FILE_PATHS["spam_list"],
        help=f"스팸 리스트 엑셀 파일 경로 (기본값: {FILE_PATHS['spam_list']})"
    )
    
    # 사용 가능한 LLM 유형 결정
    available_llm_types = ["local_ai", "exaone"]
    if 'OPENAI_AVAILABLE' in globals() and OPENAI_AVAILABLE:
        available_llm_types.append("openai")
    if 'ANTHROPIC_AVAILABLE' in globals() and ANTHROPIC_AVAILABLE:
        available_llm_types.append("anthropic")
    if 'BEDROCK_AVAILABLE' in globals() and BEDROCK_AVAILABLE:
        available_llm_types.append("claude-bedrock")
    
    # 선택적 인수
    parser.add_argument(
        "--llm", "-l",
        type=str,
        nargs="+",  # 여러 LLM 유형을 받을 수 있도록 수정
        choices=available_llm_types,
        default=["local_ai"] if "local_ai" in available_llm_types else [available_llm_types[0]] if available_llm_types else None,
        help=f"사용할 LLM 유형 (여러 개 지정 가능, 기본값: local_ai)"
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
    
    parser.add_argument(
        "--no-git", "-ng",
        action="store_true",
        help="Git 브랜치 생성 및 푸시 비활성화"
    )
    
    return parser.parse_args()

def setup_output_folder(output_path=None):
    """결과를 저장할 폴더를 설정합니다."""
    if output_path is None:
        now = datetime.now()
        output_path = f"spam_llm_classification_{now.strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_path, exist_ok=True)
    return output_path

def display_settings(args, output_folder, branch_name=None):
    """현재 설정을 표시합니다."""
    print("\n=== 스팸 분류 설정 ===")
    print(f"파일 경로: {args.file}")
    print(f"LLM 유형: {', '.join(args.llm)}")
    
    # 각 LLM 유형별 모델 정보 표시
    for llm_type in args.llm:
        if llm_type == "local_ai":
            from config import LOCAL_AI_SETTINGS
            print(f"- Local AI 기본 URL: {LOCAL_AI_SETTINGS['base_url']}")
            print(f"- Local AI 모델: {LOCAL_AI_SETTINGS['model']}")
        else:
            print(f"- {llm_type.upper()} 모델: {LLM_SETTINGS[llm_type]['model']}")
    
    if args.sample == 0:
        print("샘플 크기: 전체 데이터")
    else:
        print(f"샘플 크기: {args.sample}")
    
    print(f"결과 저장 경로: {output_folder}")
    
    if branch_name and not args.no_git:
        print(f"Git 브랜치: {branch_name}")
    elif args.no_git:
        print("Git 브랜치: 비활성화됨")
        
    print("=====================\n")

def run_git_command(command, error_message=None):
    """Git 명령어를 실행합니다."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if error_message:
            print(f"{error_message}: {e.stderr}")
        return False, e.stderr

def create_git_branch(branch_name):
    """새로운 Git 브랜치를 생성합니다."""
    # 현재 브랜치 확인
    success, current_branch = run_git_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "현재 브랜치를 확인하는 중 오류가 발생했습니다"
    )
    
    if not success:
        return False, current_branch
    
    current_branch = current_branch.strip()
    
    # 새 브랜치 생성
    success, output = run_git_command(
        ["git", "checkout", "-b", branch_name],
        f"브랜치 '{branch_name}' 생성 중 오류가 발생했습니다"
    )
    
    if not success:
        return False, output
    
    print(f"브랜치 '{branch_name}'가 생성되었습니다.")
    return True, current_branch

def push_results_to_git(output_folder, branch_name):
    """결과를 Git에 푸시합니다."""
    # 변경 사항 스테이징
    success, _ = run_git_command(
        ["git", "add", output_folder],
        f"폴더 '{output_folder}'를 스테이징하는 중 오류가 발생했습니다"
    )
    
    if not success:
        return False
    
    # 변경 사항 커밋
    commit_message = f"스팸 분류 결과 추가 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    success, _ = run_git_command(
        ["git", "commit", "-m", commit_message],
        "변경 사항을 커밋하는 중 오류가 발생했습니다"
    )
    
    if not success:
        return False
    
    # 원격 저장소에 푸시
    success, _ = run_git_command(
        ["git", "push", "-u", "origin", branch_name],
        f"브랜치 '{branch_name}'를 원격 저장소에 푸시하는 중 오류가 발생했습니다"
    )
    
    if not success:
        return False
    
    print(f"결과가 브랜치 '{branch_name}'에 성공적으로 푸시되었습니다.")
    return True

def main():
    """메인 함수"""
    # 명령줄 인수 파싱
    args = parse_arguments()
    
    # 현재 시간으로 브랜치 이름 생성
    now = datetime.now()
    branch_name = f"{now.strftime('%Y%m%d_%H%M')}"
    
    # 샘플 크기 처리 (0은 전체 데이터 사용)
    sample_size = None if args.sample == 0 else args.sample
    
    # 결과 폴더 설정
    output_folder = setup_output_folder(args.output)
    
    # Git 브랜치 생성 (--no-git 옵션이 없는 경우)
    original_branch = None
    if not args.no_git:
        success, result = create_git_branch(branch_name)
        if success:
            original_branch = result
        else:
            print(f"경고: Git 브랜치 생성에 실패했습니다. 기본 브랜치에서 계속합니다.")
    
    # 설정 표시
    display_settings(args, output_folder, branch_name)
    
    # 프롬프트 히스토리 파일 생성
    prompt_history_file = os.path.join(output_folder, "prompt_history.txt")
    with open(prompt_history_file, "w", encoding="utf-8") as f:
        f.write("# 스팸 리스트 LLM 분류 프롬프트 히스토리\n\n")
        f.write(f"## 분석 시작 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"### 명령줄 인수:\n")
        f.write(f"- 파일 경로: {args.file}\n")
        f.write(f"- LLM 유형: {', '.join(args.llm)}\n")
        f.write(f"- 샘플 크기: {'전체 데이터' if sample_size is None else sample_size}\n")
        f.write(f"- 결과 저장 경로: {output_folder}\n")
        if not args.no_git:
            f.write(f"- Git 브랜치: {branch_name}\n")
        f.write("\n")
    
    try:
        # 파일 경로 처리 - data/ 접두사 제거
        file_path = args.file
        if file_path.startswith("data/"):
            file_path = file_path.replace("data/", "", 1)
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {args.file}")
        
        # 각 LLM 유형별로 스팸 분류 실행
        all_results = {}
        for llm_type in args.llm:
            print(f"\n{llm_type.upper()} 모델로 스팸 분류 시작...")
            
            # LLM별 결과 폴더 생성
            llm_output_folder = os.path.join(output_folder, llm_type)
            os.makedirs(llm_output_folder, exist_ok=True)
            
            # 스팸 분류 실행
            df = classify_spam_messages(file_path, llm_type, sample_size)
            
            # 분류 결과 분석 및 시각화
            print(f"{llm_type.upper()} 모델 분류 결과 분석 중...")
            analyze_classification_results(df, llm_output_folder, llm_type)
            
            # 결과 저장
            all_results[llm_type] = df
            
            # 프롬프트 히스토리 업데이트
            with open(prompt_history_file, "a", encoding="utf-8") as f:
                f.write(f"### {llm_type.upper()} 모델 스팸 분류 완료\n")
                f.write(f"- 분석된 메시지 수: {len(df)}\n")
                
                if llm_type == "local_ai":
                    from config import LOCAL_AI_SETTINGS
                    f.write(f"- 사용된 모델: {LOCAL_AI_SETTINGS['model']}\n")
                else:
                    f.write(f"- 사용된 모델: {LLM_SETTINGS[llm_type]['model']}\n")
                    
                f.write(f"- 결과 저장 위치: {os.path.abspath(llm_output_folder)}\n\n")
        
        # 모든 LLM 결과 비교 분석 (2개 이상의 LLM을 사용한 경우)
        if len(args.llm) > 1:
            print("\n여러 LLM 모델 결과 비교 분석 중...")
            compare_output_folder = os.path.join(output_folder, "comparison")
            os.makedirs(compare_output_folder, exist_ok=True)
            
            # 결과 비교 분석 함수 호출 (이 함수는 별도로 구현 필요)
            compare_llm_results(all_results, compare_output_folder)
            
            # 프롬프트 히스토리 업데이트
            with open(prompt_history_file, "a", encoding="utf-8") as f:
                f.write(f"### LLM 모델 결과 비교 분석 완료\n")
                f.write(f"- 비교된 모델: {', '.join(args.llm)}\n")
                f.write(f"- 결과 저장 위치: {os.path.abspath(compare_output_folder)}\n\n")
        
        # Git에 결과 푸시 (--no-git 옵션이 없는 경우)
        if not args.no_git:
            print(f"Git에 결과 푸시 중...")
            push_success = push_results_to_git(output_folder, branch_name)
            if push_success:
                print(f"결과가 브랜치 '{branch_name}'에 성공적으로 푸시되었습니다.")
            else:
                print(f"경고: Git 푸시에 실패했습니다.")
        
        print(f"\n분석이 완료되었습니다. 결과는 '{output_folder}' 폴더에 저장되었습니다.")
        
        # 결과 요약 표시
        if args.verbose:
            for llm_type in args.llm:
                summary_file = os.path.join(output_folder, llm_type, "classification_summary.txt")
                if os.path.exists(summary_file):
                    with open(summary_file, "r", encoding="utf-8") as f:
                        print(f"\n=== {llm_type.upper()} 모델 분류 결과 요약 ===")
                        print(f.read())
                else:
                    print(f"\n{llm_type.upper()} 모델 요약 파일을 찾을 수 없습니다.")
    
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
    
    finally:
        # 원래 브랜치로 돌아가기 (--no-git 옵션이 없고 원래 브랜치가 있는 경우)
        if not args.no_git and original_branch:
            print(f"원래 브랜치 '{original_branch}'로 돌아갑니다...")
            run_git_command(
                ["git", "checkout", original_branch],
                f"브랜치 '{original_branch}'로 돌아가는 중 오류가 발생했습니다"
            )
    
    return 0

# LLM 결과 비교 분석 함수 추가
def compare_llm_results(all_results, output_folder):
    """
    여러 LLM 모델의 분류 결과를 비교 분석합니다.
    
    Args:
        all_results: 각 LLM 모델별 분류 결과 데이터프레임을 담은 딕셔너리
        output_folder: 비교 결과를 저장할 폴더 경로
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 결과 비교를 위한 데이터프레임 생성
    comparison_data = []
    
    # 기준 데이터프레임 (첫 번째 LLM 결과)
    base_df = next(iter(all_results.values()))
    
    # 각 메시지별로 LLM 분류 결과 비교
    for i, row in base_df.iterrows():
        message_id = row.get("일련번호", i)
        message_content = row.get("내용", "")
        
        # 각 LLM 모델별 분류 결과 수집
        result_row = {
            "메시지_ID": message_id,
            "내용": message_content[:100] + "..." if len(message_content) > 100 else message_content
        }
        
        # 원본 데이터의 휴먼 분류 추가 (스팸여부, 스팸유형 컬럼이 있다고 가정)
        if "스팸여부" in row:
            result_row["휴먼_is_spam"] = row["스팸여부"] == "스팸"
        elif "스팸 여부" in row:
            result_row["휴먼_is_spam"] = row["스팸 여부"] == "스팸"
        
        if "스팸유형" in row:
            result_row["휴먼_category"] = row["스팸유형"]
        elif "스팸 유형" in row:
            result_row["휴먼_category"] = row["스팸 유형"]
        
        # 각 LLM 모델의 분류 결과 추가
        for llm_type, df in all_results.items():
            if i in df.index:
                result_row[f"{llm_type}_is_spam"] = df.at[i, "llm_is_spam"]
                result_row[f"{llm_type}_category"] = df.at[i, "llm_category"]
                result_row[f"{llm_type}_confidence"] = df.at[i, "llm_confidence"]
                
                # 모델별 reason 필드 추가
                if llm_type == "openai" and "gpt_reason" in df.columns:
                    result_row[f"{llm_type}_reason"] = df.at[i, "gpt_reason"]
                elif llm_type == "anthropic":
                    if "claude_reason" in df.columns:
                        result_row[f"{llm_type}_reason"] = df.at[i, "claude_reason"]
                    elif "exaone_reason" in df.columns:
                        result_row[f"{llm_type}_reason"] = df.at[i, "exaone_reason"]
                elif llm_type == "claude-bedrock" and "claude_reason" in df.columns:
                    result_row[f"{llm_type}_reason"] = df.at[i, "claude_reason"]
                elif llm_type == "exaone" and "exaone_reason" in df.columns:
                    result_row[f"{llm_type}_reason"] = df.at[i, "exaone_reason"]
                elif "local_ai_reason" in df.columns:
                    result_row[f"{llm_type}_reason"] = df.at[i, "local_ai_reason"]
        
        comparison_data.append(result_row)
    
    # 비교 데이터프레임 생성
    comparison_df = pd.DataFrame(comparison_data)
    
    # 결과 저장
    comparison_df.to_csv(os.path.join(output_folder, "llm_comparison_results.csv"), index=False, encoding="utf-8-sig")
    
    # 휴먼 분류와 LLM 분류 비교 분석
    human_comparison_data = []
    
    # 휴먼 분류가 있는 경우에만 비교 분석 수행
    if "휴먼_is_spam" in comparison_df.columns:
        for llm_type in all_results.keys():
            # 스팸 여부 일치율
            spam_agreement = (comparison_df["휴먼_is_spam"] == comparison_df[f"{llm_type}_is_spam"]).mean()
            
            # 카테고리 일치율 (휴먼 카테고리가 있는 경우)
            category_agreement = 0.0
            if "휴먼_category" in comparison_df.columns:
                # 스팸으로 분류된 항목 중에서만 카테고리 일치율 계산
                spam_items = comparison_df[comparison_df["휴먼_is_spam"] & comparison_df[f"{llm_type}_is_spam"]]
                if len(spam_items) > 0:
                    category_agreement = (spam_items["휴먼_category"] == spam_items[f"{llm_type}_category"]).mean()
            
            human_comparison_data.append({
                "LLM": llm_type,
                "휴먼_스팸여부_일치율": spam_agreement,
                "휴먼_카테고리_일치율": category_agreement
            })
        
        # 휴먼 비교 데이터프레임 생성
        human_comparison_df = pd.DataFrame(human_comparison_data)
        human_comparison_df.to_csv(os.path.join(output_folder, "human_vs_llm_agreement.csv"), index=False, encoding="utf-8-sig")
        
        # 휴먼 vs LLM 일치율 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(x="LLM", y="휴먼_스팸여부_일치율", data=human_comparison_df)
        plt.title("휴먼 vs LLM 스팸 여부 일치율")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "human_vs_llm_spam_agreement.png"))
        plt.close()
        
        if "휴먼_category" in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="LLM", y="휴먼_카테고리_일치율", data=human_comparison_df)
            plt.title("휴먼 vs LLM 카테고리 일치율")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "human_vs_llm_category_agreement.png"))
            plt.close()
    
    # LLM 간 일치율 분석
    if len(all_results) > 1:
        llm_types = list(all_results.keys())
        agreement_data = []
        
        # 모든 LLM 쌍에 대해 일치율 계산
        for i in range(len(llm_types)):
            for j in range(i+1, len(llm_types)):
                llm1 = llm_types[i]
                llm2 = llm_types[j]
                
                # 스팸 여부 일치율
                spam_agreement = (comparison_df[f"{llm1}_is_spam"] == comparison_df[f"{llm2}_is_spam"]).mean()
                
                # 카테고리 일치율
                category_agreement = (comparison_df[f"{llm1}_category"] == comparison_df[f"{llm2}_category"]).mean()
                
                agreement_data.append({
                    "LLM1": llm1,
                    "LLM2": llm2,
                    "스팸_여부_일치율": spam_agreement,
                    "카테고리_일치율": category_agreement
                })
        
        # 일치율 데이터프레임 생성
        agreement_df = pd.DataFrame(agreement_data)
        agreement_df.to_csv(os.path.join(output_folder, "llm_agreement_rates.csv"), index=False, encoding="utf-8-sig")
        
        # 일치율 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(x="LLM1", y="스팸_여부_일치율", hue="LLM2", data=agreement_df)
        plt.title("LLM 모델 간 스팸 여부 일치율")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "spam_agreement_rates.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="LLM1", y="카테고리_일치율", hue="LLM2", data=agreement_df)
        plt.title("LLM 모델 간 카테고리 일치율")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "category_agreement_rates.png"))
        plt.close()
    
    # 요약 정보 저장
    with open(os.path.join(output_folder, "comparison_summary.txt"), "w", encoding="utf-8") as f:
        f.write("# LLM 모델 비교 분석 결과 요약\n\n")
        f.write(f"## 비교된 LLM 모델\n")
        for llm_type in all_results.keys():
            if llm_type == "local_ai":
                from config import LOCAL_AI_SETTINGS
                f.write(f"- {llm_type.upper()}: {LOCAL_AI_SETTINGS['model']}\n")
            else:
                f.write(f"- {llm_type.upper()}: {LLM_SETTINGS[llm_type]['model']}\n")
        
        f.write(f"\n## 분석된 메시지 수: {len(comparison_df)}\n\n")
        
        # 휴먼 vs LLM 일치율 요약
        if "휴먼_is_spam" in comparison_df.columns:
            f.write(f"## 휴먼 vs LLM 일치율\n")
            for _, row in human_comparison_df.iterrows():
                f.write(f"- 휴먼 vs {row['LLM'].upper()}:\n")
                f.write(f"  - 스팸 여부 일치율: {row['휴먼_스팸여부_일치율']:.2%}\n")
                if "휴먼_category" in comparison_df.columns:
                    f.write(f"  - 카테고리 일치율: {row['휴먼_카테고리_일치율']:.2%}\n")
        
        # LLM 간 일치율 요약
        if len(all_results) > 1:
            f.write(f"\n## LLM 모델 간 일치율\n")
            for _, row in agreement_df.iterrows():
                f.write(f"- {row['LLM1'].upper()} vs {row['LLM2'].upper()}:\n")
                f.write(f"  - 스팸 여부 일치율: {row['스팸_여부_일치율']:.2%}\n")
                f.write(f"  - 카테고리 일치율: {row['카테고리_일치율']:.2%}\n")
        
        # 각 LLM 모델별 스팸 비율
        f.write(f"\n## 각 LLM 모델별 스팸 비율\n")
        for llm_type in all_results.keys():
            spam_ratio = comparison_df[f"{llm_type}_is_spam"].mean()
            f.write(f"- {llm_type.upper()}: {spam_ratio:.2%}\n")
        
        # 휴먼 분류 스팸 비율
        if "휴먼_is_spam" in comparison_df.columns:
            human_spam_ratio = comparison_df["휴먼_is_spam"].mean()
            f.write(f"- 휴먼 분류: {human_spam_ratio:.2%}\n")
    
    print(f"LLM 모델 비교 분석이 완료되었습니다. 결과는 '{output_folder}' 폴더에 저장되었습니다.")
    return comparison_df

if __name__ == "__main__":
    sys.exit(main()) 