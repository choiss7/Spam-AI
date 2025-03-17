#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GitHub API를 사용하여 레포지토리 목록을 가져오는 스크립트
"""

import requests
import json

def get_repositories(token):
    """GitHub API를 사용하여 레포지토리 목록을 가져옵니다."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = "https://api.github.com/user/repos?per_page=100"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            repos = response.json()
            print(f"총 {len(repos)}개의 레포지토리를 찾았습니다.\n")
            
            print("레포지토리 목록:")
            for i, repo in enumerate(repos, 1):
                print(f"{i}. {repo['full_name']} - {repo['html_url']}")
                print(f"   설명: {repo['description'] or '설명 없음'}")
                print(f"   언어: {repo['language'] or '언어 정보 없음'}")
                print(f"   생성일: {repo['created_at']}")
                print(f"   최근 업데이트: {repo['updated_at']}")
                print(f"   스타: {repo['stargazers_count']}, 포크: {repo['forks_count']}")
                print()
        else:
            print(f"오류 발생: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # GitHub 개인 액세스 토큰
    token = "YOUR_TOKEN_HERE"
    
    get_repositories(token) 