#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
필요한 패키지 설치 스크립트
"""

import subprocess
import sys
import os

def install_package(package):
    """패키지를 설치합니다."""
    try:
        print(f"{package} 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 설치 완료!")
        return True
    except subprocess.CalledProcessError:
        print(f"{package} 설치 실패!")
        return False

def main():
    """메인 함수"""
    print("스팸 분류기에 필요한 패키지를 설치합니다.")
    
    # 필수 패키지 목록
    required_packages = [
        "pandas",
        "openpyxl",
        "matplotlib",
        "seaborn",
        "python-dotenv",
        "requests"
    ]
    
    # 선택적 패키지 목록
    optional_packages = [
        "openai",
        "anthropic"
    ]
    
    # 필수 패키지 설치
    print("\n=== 필수 패키지 설치 ===")
    for package in required_packages:
        install_package(package)
    
    # 선택적 패키지 설치
    print("\n=== 선택적 패키지 설치 ===")
    print("다음 패키지는 선택적으로 설치할 수 있습니다.")
    
    for package in optional_packages:
        install = input(f"{package}를 설치하시겠습니까? (y/n): ").lower().strip()
        if install == 'y':
            install_package(package)
        else:
            print(f"{package} 설치를 건너뜁니다.")
    
    print("\n모든 패키지 설치가 완료되었습니다.")
    print("이제 'python spam_classifier_cli.py --llm local_ai' 명령으로 스팸 분류기를 실행할 수 있습니다.")

if __name__ == "__main__":
    main() 