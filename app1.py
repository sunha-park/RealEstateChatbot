from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import calendar
from datetime import datetime
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import json
import mysql.connector
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import KDTree #KD트리
from transformers import pipeline
import openai
import numpy as np
from faiss_gpt import QnA_with_RAG

app = Flask(__name__)
app.secret_key = 'sunha'  # 세션 사용을 위한 secret key 설정

# MySQL 연결 설정
db_config = {
    'user': 'sunha',
    'password': '1234',
    'host': 'localhost',
    'database': 'backend'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)


# -------------------------------------------------------------------------------부동산 매물 확인
# Load data
file_path = '아파트실거래가.csv'
df = pd.read_csv(file_path, encoding='cp949')

# Process and clean data
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)
df['계약년월'] = pd.to_datetime(df['계약년월'], format='%Y%m')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

def apply_filters(data, floor=None, area=None):
    if floor == '지상층':
        data = data[data['층'] > 0]
    elif floor == '반지하':
        data = data[data['층'] < 0]
    if area:
        area_mapping = {
            '10평 이하': (0, 33), '10평대': (33, 66), '20평대': (66, 99),
            '30평대': (99, 132), '40평대': (132, 165), '50평대': (165, 198),
            '60평 이상': (198, float('inf'))
        }
        min_area, max_area = area_mapping.get(area, (0, float('inf')))
        data = data[(data['전용면적(㎡)'] >= min_area) & (data['전용면적(㎡)'] < max_area)]

    return data
 

def generate_map(data):
    folium_map = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
    marker_cluster = MarkerCluster().add_to(folium_map)

    # 매물 정보를 가진 마커 생성
    for _, row in data.iterrows():
        popup_html = f"""
            <div style="
                color: #333; 
                width: 200px; 
                padding: 5px; 
                border-radius: 8px;
                background-color: #fff;">
                
                <h4 style="color: #2c3e50; margin: 0 0 10px;">{row['단지명']}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            <strong>전용면적</strong>
                        </td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            {row['전용면적(㎡)']}㎡
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            <strong>거래금액</strong>
                        </td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            {row['거래금액(만원)']}만원
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            <strong>층</strong>
                        </td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            {row['층']}층
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            <strong>건축년도</strong>
                        </td>
                        <td style="padding: 4px 8px; border-bottom: 1px solid #ddd;">
                            {row['건축년도']}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 8px;">
                            <strong>주소</strong>
                        </td>
                        <td style="padding: 4px 8px;">
                            {row['도로명']}
                        </td>
                    </tr>
                </table>
            </div>
        """
        
        # 마커 추가
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(marker_cluster)

    return folium_map


@app.route('/')
def index():
    # 전체 매물 지도를 생성
    folium_map = generate_map(df)
    folium_map.save('static/map.html')
    return render_template('index1.html', map_file="map.html")


@app.route('/search', methods=['GET'])
def search():
    keyword = request.args.get('keyword', '').strip()
    floor = request.args.get('floor')
    area = request.args.get('area')
    min_price = request.args.get('min_price', type=int)
    max_price = request.args.get('max_price', type=int)

    filtered_df = df.copy()
    if keyword:
        filtered_df = filtered_df[filtered_df['단지명'].str.contains(keyword, case=False, na=False) |
                                  filtered_df['도로명'].str.contains(keyword, case=False, na=False)]

    filtered_df = apply_filters(filtered_df, floor, area)

    folium_map = generate_map(filtered_df)
    folium_map.save('static/map.html')

    return render_template('index1.html', map_file="map.html", keyword=keyword)



# ---------------------------------------------------------------------------------------------메모장
# 임시 데이터 저장소
tasks = {}

def generate_calendar():
    """현재 월의 캘린더 데이터를 반환합니다."""
    today = datetime.today()
    year, month = today.year, today.month
    cal = calendar.Calendar(firstweekday=6)  # 일요일을 첫 번째 요일로 설정
    month_days = cal.monthdayscalendar(year, month)
    return year, month, month_days


# 분석메모 페이지
@app.route('/memo')
def memo():
    year, month, month_days = generate_calendar()
    return render_template('memo.html', tasks=tasks, year=year, month=month, month_days=month_days)

@app.route('/add', methods=['POST'])
def add_task():
    task_content = request.form.get('task')
    task_date = request.form.get('date')
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if task_content and task_date:
        if task_date not in tasks:
            tasks[task_date] = []
        tasks[task_date].append({
            'content': task_content,
            'done': False,
            'latitude': latitude,
            'longitude': longitude
        })

    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("INSERT INTO note (task_date, task_content, latitude, longitude) VALUES (%s, %s, %s, %s)", 
                   (task_date, task_content, latitude, longitude))
    db.commit()

    # 생성된 사용자 ID를 세션에 저장
    user_id = cursor.lastrowid
    session['user_id'] = user_id

    cursor.close()
    db.close()

    return redirect(url_for('memo'))

@app.route('/toggle/<task_date>/<int:task_id>')
def toggle_task(task_date, task_id):
    if task_date in tasks and 0 <= task_id < len(tasks[task_date]):
        tasks[task_date][task_id]['done'] = True
    return redirect(url_for('memo'))

@app.route('/delete/<task_date>/<int:task_id>')
def delete_task(task_date, task_id):
    # 해당 날짜와 ID가 존재하는지 확인
    if task_date in tasks and 0 <= task_id < len(tasks[task_date]):
        # 해당 ID의 일정 삭제
        tasks[task_date].pop(task_id)
        
        # 만약 해당 날짜의 모든 일정이 삭제되었다면, 날짜 자체를 삭제
        if not tasks[task_date]:  # 날짜의 일정 리스트가 비어 있으면
            del tasks[task_date]
        
        # 삭제 성공 시, 성공 상태와 함께 갱신된 할 일 목록 반환
        return jsonify(status="success", tasks=tasks)
    else:
        # 삭제할 항목이 없으면 에러 상태 반환
        return jsonify(status="error", message="삭제할 항목을 찾을 수 없습니다."), 404

@app.route('/save_note', methods=['POST'])
def save_note():
    # JSON 데이터가 아닌 경우 오류 반환
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Invalid content type'}), 415

    data = request.get_json()
    memo_content = data.get('content')

    user_id = session.get('user_id')
    db = get_db_connection()
    cursor = db.cursor()

    if user_id:
        cursor.execute("UPDATE note SET memo_content = %s WHERE id = %s", (memo_content, user_id))
    else:
        cursor.execute("INSERT INTO note (memo_content) VALUES (%s)", (memo_content,))
        session['user_id'] = cursor.lastrowid

    db.commit()
    cursor.close()
    db.close()

    return jsonify({'status': 'success', 'memo_content': memo_content})

@app.route('/get_note', methods=['GET'])
def get_note():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User ID not found in session'})

    # 메모 내용을 DB에서 가져옵니다
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT memo_content FROM note WHERE id = %s", (user_id,))
    memo_content = cursor.fetchone()
    cursor.close()
    db.close()

    if memo_content:
        return jsonify({'status': 'success', 'memo_content': memo_content[0]})
    else:
        return jsonify({'status': 'error', 'message': 'No memo found'})


@app.route('/calendar/<int:year>/<int:month>')
def get_calendar(year, month):
    cal = calendar.Calendar(firstweekday=6)  # 일요일부터 시작
    month_days = cal.monthdayscalendar(year, month)
    
    # 날짜별로 할 일 데이터를 추가하여 반환
    tasks_for_month = {date: tasks[date] for date in tasks if date.startswith(f"{year}-{month:02d}")}
    
    return jsonify({
        'year': year,
        'month': month,
        'month_days': month_days,
        'tasks': tasks_for_month  # 해당 월의 할 일만 전달
    })


#------------------------------------------------------------------------------------------------------------분석하기
# OpenAI API 키 설정
openai.api_key = ""

bicycle = pd.read_csv('공공자전거 대여소.csv', encoding='cp949')
station = pd.read_csv('서울지하철역.csv', encoding='cp949')
bus = pd.read_csv('버스정류소.csv', encoding='cp949')

def calculate_latlon(df,a_lat,a_lon):
    # 위도와 경도 정보를 이용해 KD 트리 생성
    tree = KDTree(df[['위도', '경도']])

    # 최근접 이웃 검색 (여기서는 A편의점과 가장 가까운 이웃 1개만 찾음)
    distance, index = tree.query([[a_lat, a_lon]], k=1)

    return distance[0]

def get_user_location(task_date, task_content, memo_content):
    db = get_db_connection()
    cursor = db.cursor()
    query = """
        SELECT latitude, longitude 
        FROM note 
        WHERE task_date = %s AND task_content = %s AND memo_content = %s
    """
    cursor.execute(query, (task_date, task_content, memo_content))
    result = cursor.fetchone()
    cursor.close()
    db.close()
    if result:
        lat, lon = result
        return lat, lon
    return None, None

def generate_response(bicycle_distance, station_distance, bus_distance, memo_content, result):
    
    # LLM을 통해 요약된 데이터를 바탕으로 응답 생성
    prompt = f"""
    chatGPT 너는 공인중개사야, 집 매물과의 따릉이 대여소 거리 차이는 {bicycle_distance}이고,
    지하철 역까지의 거리는 {station_distance}이고, 버스 정류장까지의 거리는 {bus_distance}야.
    이걸 요약해줘.
    한편, 집 매물을 본 고객이 {memo_content} 내용과 같은 후기를 남겼고 감정분석 결과는 {result}와 같아.
    이에 대해 고객에게 자세한 상담을 해줘.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    return response['choices'][0]['message']['content']

@app.route('/analysis')
def analysis():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # task_date, task_content, memo_content가 NULL이 아닌 행만 조회
    query = """
        SELECT task_date, task_content, memo_content 
        FROM note 
        WHERE task_date IS NOT NULL 
          AND task_content IS NOT NULL 
          AND memo_content IS NOT NULL
    """
    cursor.execute(query)
    notes = cursor.fetchall()
    
    cursor.close()
    db.close()

    return render_template('analysis.html', notes=notes)

@app.route('/delete_note', methods=['DELETE'])
def delete_note():
    task_date = request.args.get('task_date')
    task_content = request.args.get('task_content')
    memo_content = request.args.get('memo_content')

    db = get_db_connection()
    cursor = db.cursor()

    # note 테이블에서 주어진 task_date, task_content, memo_content에 해당하는 노트를 삭제
    cursor.execute(
        "DELETE FROM note WHERE task_date = %s AND task_content = %s AND memo_content = %s",
        (task_date, task_content, memo_content)
    )
    db.commit()

    cursor.close()
    db.close()
    
    return jsonify({"status": "success"}), 200





@app.route('/note_detail')
def note_detail():
    # URL 파라미터로 받은 노트 정보를 가져옴
    task_date = request.args.get('task_date')
    task_content = request.args.get('task_content')
    memo_content = request.args.get('memo_content')

    # 감정 분석 파이프라인 초기화
    nlp = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # 감정 분석 수행
    sentiment = nlp(memo_content)

    # 결과 확인 및 출력
    if sentiment[0]['label'] == '5 stars':
        result = "★★★★★"
    elif sentiment[0]['label'] == '4 stars':
        result = "★★★★"
    elif sentiment[0]['label'] == '3 stars':
        result = "★★★"
    elif sentiment[0]['label'] == '2 stars':
        result = "★★"
    elif sentiment[0]['label'] == '1 star':
        result = "★"
    else:
        result = "알 수 없음"

    lat_a, lon_a = get_user_location(task_date, task_content, memo_content)  # note 테이블에서 사용자 위치 가져오기
    # 사용자 위치가 유효한 경우에만 계산 수행
    if lat_a is not None and lon_a is not None:
        bicycle_distance = calculate_latlon(bicycle, lat_a, lon_a)
        station_distance = calculate_latlon(station, lat_a, lon_a)
        bus_distance = calculate_latlon(bus, lat_a, lon_a)
    else:
        return "사용자의 위치 정보를 찾을 수 없습니다.", 400

    advice = generate_response(bicycle_distance, station_distance, bus_distance, memo_content, result)
    # 가져온 정보를 새로운 템플릿에 전달하여 화면에 표시
    return render_template('note_detail.html', task_date=task_date,
                                               task_content=task_content,
                                               memo_content=memo_content,
                                               bicycle_distance=round(bicycle_distance*1000,2), 
                                               station_distance=round(station_distance*1000,2), 
                                               bus_distance=round(bus_distance*1000,2),
                                               result=result,
                                               advice=advice)


#-----------------------------------------------------------------------------------------------------------챗봇
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('chat.html')  # GET 요청 시 chat.html 반환
    elif request.method == 'POST':
        data = request.get_json()
        user_message = data.get('message', '')
        keyword = data.get('keyword', '')

        # llm.py의 generate_response 함수 호출
        if user_message:  # 메시지가 있는 경우만 처리
            response_message = QnA_with_RAG(f"{keyword}에 관한 질문: {user_message}")
            print(f"{keyword}에 관한 질문: {user_message}")
        else:
            response_message = "질문을 입력해주세요."  # 사용자 메시지가 없는 경우 기본 응답

        return jsonify({'response': response_message})  # POST 요청에 대한 JSON 응답



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
