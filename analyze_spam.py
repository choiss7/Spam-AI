import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 한글 폰트 설정 (윈도우 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 현재 시간을 기반으로 결과 폴더 생성
now = datetime.now()
result_folder = f"spam_analysis_{now.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(result_folder, exist_ok=True)

# 프롬프트 히스토리 파일 생성 및 초기화
prompt_history_file = os.path.join(result_folder, "prompt_history.txt")
with open(prompt_history_file, "w", encoding="utf-8") as f:
    f.write("# 스팸 리스트 분석 프롬프트 히스토리\n\n")
    f.write(f"## 분석 시작 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("### 프롬프트: 프로젝트 폴더에 있는 스팸 리스트 엑셀 파일을 분석해 주세요\n\n")

# 엑셀 파일 읽기
file_path = "스팸리스트_20250310_개인정보_삭제.xlsx"
print(f"파일 '{file_path}' 읽는 중...")

try:
    df = pd.read_excel(file_path)
    
    # 데이터 기본 정보 출력 및 저장
    print(f"데이터 크기: {df.shape}")
    
    # 기본 정보 저장
    with open(os.path.join(result_folder, "basic_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"# 스팸 리스트 기본 정보\n\n")
        f.write(f"## 데이터 크기: {df.shape}\n\n")
        f.write("## 컬럼 목록:\n")
        for col in df.columns:
            f.write(f"- {col}\n")
        
        f.write("\n## 데이터 타입:\n")
        f.write(str(df.dtypes))
        
        f.write("\n\n## 기본 통계:\n")
        f.write(str(df.describe()))
        
        f.write("\n\n## 결측치 정보:\n")
        f.write(str(df.isnull().sum()))
    
    # 프롬프트 히스토리 업데이트
    with open(prompt_history_file, "a", encoding="utf-8") as f:
        f.write("### 결과: 기본 정보 분석 완료\n")
        f.write(f"- 데이터 크기: {df.shape}\n")
        f.write(f"- 컬럼 수: {len(df.columns)}\n\n")
    
    # 데이터 시각화
    print("데이터 시각화 중...")
    
    # 1. 스팸 유형 분포 (만약 해당 컬럼이 있다면)
    spam_type_col = None
    for col in df.columns:
        if '유형' in col or '종류' in col or '분류' in col or '카테고리' in col:
            spam_type_col = col
            break
    
    if spam_type_col:
        plt.figure(figsize=(12, 6))
        # 수정된 부분: 직접 value_counts를 사용하여 시각화
        value_counts = df[spam_type_col].value_counts()
        sns.barplot(x=value_counts.values, y=value_counts.index)
        plt.title(f'스팸 {spam_type_col} 분포')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, "spam_type_distribution.png"))
        plt.close()
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write(f"### 결과: 스팸 {spam_type_col} 분포 시각화 완료\n\n")
    
    # 2. 시간대별 스팸 발생 빈도 (날짜/시간 컬럼이 있다면)
    date_col = None
    for col in df.columns:
        if '날짜' in col or '일시' in col or '시간' in col or 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col:
        # 날짜 형식으로 변환 시도
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df['hour'] = df[date_col].dt.hour
            df['date'] = df[date_col].dt.date
            
            # 시간대별 분포
            plt.figure(figsize=(12, 6))
            hour_counts = df['hour'].value_counts().sort_index()
            sns.barplot(x=hour_counts.index, y=hour_counts.values)
            plt.title('시간대별 스팸 발생 빈도')
            plt.xlabel('시간')
            plt.ylabel('빈도')
            plt.xticks(range(0, 24))
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, "spam_by_hour.png"))
            plt.close()
            
            # 일자별 분포
            plt.figure(figsize=(14, 6))
            df['date'].value_counts().sort_index().plot(kind='line')
            plt.title('일자별 스팸 발생 빈도')
            plt.xlabel('날짜')
            plt.ylabel('빈도')
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, "spam_by_date.png"))
            plt.close()
            
            # 프롬프트 히스토리 업데이트
            with open(prompt_history_file, "a", encoding="utf-8") as f:
                f.write("### 결과: 시간대별 및 일자별 스팸 발생 빈도 시각화 완료\n\n")
        except Exception as e:
            print(f"'{date_col}' 컬럼을 날짜 형식으로 변환할 수 없습니다: {e}")
    
    # 3. 텍스트 데이터 분석 (텍스트 컬럼이 있다면)
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and ('내용' in col or '메시지' in col or 'text' in col.lower() or '문자' in col):
            text_cols.append(col)
    
    if text_cols:
        with open(os.path.join(result_folder, "text_analysis.txt"), "w", encoding="utf-8") as f:
            f.write("# 텍스트 데이터 분석\n\n")
            
            for col in text_cols:
                f.write(f"## {col} 컬럼 분석\n\n")
                
                # 텍스트 길이 분석
                df[f'{col}_length'] = df[col].astype(str).apply(len)
                
                f.write(f"### 텍스트 길이 통계:\n")
                f.write(str(df[f'{col}_length'].describe()))
                
                # 텍스트 길이 분포 시각화
                plt.figure(figsize=(10, 6))
                sns.histplot(df[f'{col}_length'], bins=30)
                plt.title(f'{col} 텍스트 길이 분포')
                plt.xlabel('텍스트 길이')
                plt.ylabel('빈도')
                plt.tight_layout()
                plt.savefig(os.path.join(result_folder, f"{col}_length_distribution.png"))
                plt.close()
                
                # 가장 빈번한 단어 분석 (간단한 버전)
                f.write(f"\n### 자주 등장하는 단어 (상위 20개):\n")
                
                # 모든 텍스트 합치기
                all_text = ' '.join(df[col].astype(str).tolist())
                
                # 간단한 단어 빈도 분석
                words = all_text.split()
                word_freq = {}
                for word in words:
                    if len(word) > 1:  # 한 글자 단어 제외
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # 상위 20개 단어
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                for word, freq in top_words:
                    f.write(f"- {word}: {freq}회\n")
                
                # 상위 단어 시각화
                if top_words:  # 단어가 있는 경우에만 시각화
                    plt.figure(figsize=(12, 8))
                    words, freqs = zip(*top_words)
                    sns.barplot(x=list(freqs), y=list(words))
                    plt.title(f'{col} 컬럼 자주 등장하는 단어 (상위 20개)')
                    plt.xlabel('빈도')
                    plt.tight_layout()
                    plt.savefig(os.path.join(result_folder, f"{col}_top_words.png"))
                    plt.close()
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write("### 결과: 텍스트 데이터 분석 완료\n")
            f.write(f"- 분석된 텍스트 컬럼: {', '.join(text_cols)}\n\n")
    
    # 4. 상관관계 분석 (숫자형 데이터가 있다면)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('숫자형 변수 간 상관관계')
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, "correlation_heatmap.png"))
        plt.close()
        
        # 프롬프트 히스토리 업데이트
        with open(prompt_history_file, "a", encoding="utf-8") as f:
            f.write("### 결과: 숫자형 변수 간 상관관계 분석 완료\n\n")
    
    # 5. 데이터 샘플 저장
    sample_size = min(10, len(df))
    df.head(sample_size).to_csv(os.path.join(result_folder, "data_sample.csv"), index=False, encoding='utf-8-sig')
    
    # 분석 완료 메시지
    print(f"분석이 완료되었습니다. 결과는 '{result_folder}' 폴더에 저장되었습니다.")
    
    # 프롬프트 히스토리 최종 업데이트
    with open(prompt_history_file, "a", encoding="utf-8") as f:
        f.write(f"## 분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"## 결과 저장 위치: {os.path.abspath(result_folder)}\n")

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