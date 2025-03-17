#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GitHub API를 사용하여 레포지토리의 파일 목록을 확인하는 스크립트
"""

import requests
import json

def get_repository_files(owner, repo):
    """GitHub API를 사용하여 레포지토리의 파일 목록을 가져옵니다."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            contents = response.json()
            print(f"레포지토리 {owner}/{repo}의 파일 목록:\n")
            
            for item in contents:
                if item["type"] == "file":
                    print(f"- {item['name']} ({item['size']} bytes)")
                elif item["type"] == "dir":
                    print(f"- {item['name']}/ (디렉토리)")
        else:
            print(f"오류 발생: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 레포지토리 정보
    owner = "choiss7"
    repo = "Spam-AI"
    
    get_repository_files(owner, repo) 