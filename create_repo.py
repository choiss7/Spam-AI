#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GitHub API를 사용하여 레포지토리를 생성하는 스크립트
"""

import requests
import json

def create_repository(token, name, description, private=False):
    """GitHub API를 사용하여 레포지토리를 생성합니다."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = "https://api.github.com/user/repos"
    
    data = {
        "name": name,
        "description": description,
        "private": private
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 201:
            repo = response.json()
            print(f"레포지토리가 성공적으로 생성되었습니다: {repo['html_url']}")
            return repo['clone_url']
        else:
            print(f"오류 발생: {response.status_code}")
            print(response.text)
            return None
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    # GitHub 개인 액세스 토큰
    token = "YOUR_TOKEN_HERE"
    
    # 레포지토리 정보
    name = "Spam-AI"
    description = "LLM 기반 스팸 분류 프로젝트"
    
    # 레포지토리 생성
    clone_url = create_repository(token, name, description, private=False)
    
    if clone_url:
        print(f"\n다음 명령으로 원격 저장소를 추가하세요:")
        print(f"git remote add origin {clone_url}")
        print(f"git push -u origin master") 