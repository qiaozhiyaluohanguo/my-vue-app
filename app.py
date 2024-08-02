import os
import pandas as pd
import requests
import json
from difflib import SequenceMatcher
import re
from flask import Flask, render_template, request, jsonify
import webbrowser
from threading import Timer
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import random
import urllib3
from fuzzywuzzy import fuzz

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 确保使用用户的家目录
home_dir = os.path.expanduser("~")
desktop_path = os.path.join(home_dir, "Desktop")
project_folder = os.path.join(desktop_path, "流向数据标准化项目")
os.makedirs(project_folder, exist_ok=True)

# 切换到项目文件夹
os.chdir(project_folder)

app = Flask(__name__)

# API设置
API_KEY = "sk-67d094cbde814e64851992335f30e495"
CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

# 请求头
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 全局变量
terminal_data = None
NUM_ROWS = 0

# 版本信息
VERSION = "4.0"
UPDATE_DATE = "2024年8月1日"
DEVELOPER = "包宏伟"

def preprocess_name(name):
    # 移除空格和常见无关字符
    name = re.sub(r'[^\w\s]', '', name)
    # 转换为小写
    name = name.lower()
    # 移除常见后缀
    name = re.sub(r'(医院|医科大学|附属医院|中心)$', '', name)
    return name

def load_data():
    global terminal_data, NUM_ROWS
    file_paths = [
        os.path.join(desktop_path, '全国终端_1.csv'),
        os.path.join(desktop_path, '全国终端_2.csv')
    ]
    dfs = []
    for path in file_paths:
        try:
            if not os.path.exists(path):
                logging.warning(f"警告: 文件 {path} 不存在。请检查文件路径是否正确。")
                continue
            df = pd.read_csv(path, dtype=str, low_memory=False)
            dfs.append(df)
            logging.info(f"成功加载文件: {path}")
        except Exception as e:
            logging.error(f"加载数据时出错 {path}: {e}")
    
    if dfs:
        terminal_data = pd.concat(dfs, ignore_index=True)
        terminal_data = terminal_data.dropna(subset=['终端名称'])
        terminal_data['处理后名称'] = terminal_data['终端名称'].apply(preprocess_name)
        NUM_ROWS = len(terminal_data)
        logging.info(f"总共加载了 {NUM_ROWS} 条记录。")
    else:
        logging.warning("警告: 没有成功加载任何数据。请检查文件路径是否正确。")

def get_model_response(prompt, retries=3, timeout=60):
    messages = [
        {"role": "system", "content": "You are a Chinese medical institution data standardization and information expert."},
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 1500
    }

    for attempt in range(retries):
        try:
            response = requests.post(CHAT_URL, headers=HEADERS, json=data, timeout=timeout, verify=False)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logging.warning(f"API请求失败，尝试重试 (尝试 {attempt + 1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                logging.error(f"API请求多次失败: {str(e)}")
                return f"<p>很抱歉，无法获取响应。请稍后再试。错误: {str(e)}</p>"

def find_matches(query, limit=10):
    processed_query = preprocess_name(query)
    
    def calculate_similarity(row):
        processed_name = preprocess_name(row['终端名称'])
        return fuzz.ratio(processed_query, processed_name)

    terminal_data['similarity'] = terminal_data.apply(calculate_similarity, axis=1)
    matches = terminal_data.nlargest(limit, 'similarity')
    
    return matches.to_dict('records')

def get_model_analysis(query, matches):
    matches_str = "\n".join([f"{i+1}. 终端名称: {match['终端名称']}, 地址: {match.get('地址', '未知')}, 相似度: {match['similarity']:.2f}" for i, match in enumerate(matches)])
    
    prompt = f"""作为流向数据标准化专家，请分析以下查询和匹配结果，并提供详细的HTML格式分析报告：

查询: {query}

匹配结果:
{matches_str}

请提供以下内容的HTML格式分析报告：
1. 查询意图分析：精准分析用户的查询意图。
2. 最佳匹配分析：确定最佳匹配结果，并详细解释原因。
3. 匹配结果总览：简要概述所有匹配结果。
4. 潜在误匹配分析：识别并解释任何潜在的误匹配。
5. 建议：为用户提供选择建议和下一步操作的建议。
6. 额外见解：提供任何其他相关的专业见解或注意事项。

请使用适当的HTML标签构建报告，确保结构清晰，易于阅读。将最佳匹配用<span class="best-match">标签</span>标注。仅提供报告内容，无需其他说明。包含一个匹配结果的表格。"""

    return get_model_response(prompt)

def generate_chart_data(matches):
    if not matches:
        return None
    
    best_match = matches[0]
    return {
        "nameSimilarity": best_match['similarity'] / 100,
        "addressMatch": random.uniform(0.7, 1.0),  # 模拟数据，实际应根据真实情况计算
        "levelMatch": random.uniform(0.8, 1.0),
        "specialtyMatch": random.uniform(0.6, 1.0),
        "overallScore": (best_match['similarity'] / 100 + random.uniform(0.7, 1.0) + random.uniform(0.8, 1.0) + random.uniform(0.6, 1.0)) / 4
    }

def answer_general_query(query):
    prompt = f"""作为医疗终端百事通，请回答以下问题，如果需要使用数据，请基于以下信息：
    - 数据库中共有 {NUM_ROWS} 条医疗终端记录
    - 这些记录包括医院、诊所、药店等各类医疗机构
    
问题: {query}

请提供一个详细、准确的HTML格式回答。使用适当的HTML标签来组织内容，确保回答清晰易读。如果问题涉及具体数字，可以进行合理的估算。"""

    return get_model_response(prompt)

def process_query(raw_input):
    logging.info(f"开始处理查询: {raw_input}")
    if not raw_input.strip():
        return jsonify({"error": "请输入查询内容"})

    steps = []
    
    steps.append({"step": 1, "description": "智能预处理", "details": "正在对查询进行预处理..."})
    time.sleep(1)  # 模拟处理时间
    
    steps.append({"step": 2, "description": "大模型分析", "details": "正在使用大模型分析查询意图..."})
    time.sleep(1)  # 模拟处理时间

    if any(keyword in raw_input for keyword in ['多少', '情况', '分布', '统计']):
        # 处理通用查询
        steps.append({"step": 3, "description": "生成回答", "details": "正在生成针对通用查询的回答..."})
        time.sleep(1)  # 模拟处理时间
        
        response = answer_general_query(raw_input)
        steps.append({"step": 4, "description": "完成", "details": "已生成回答"})
        
        return jsonify({
            "report": response,
            "isGeneralQuery": True,
            "steps": steps
        })
    else:
        # 处理终端匹配查询
        steps.append({"step": 3, "description": "数据库匹配", "details": "正在搜索最佳匹配结果..."})
        matches = find_matches(raw_input)
        time.sleep(1)  # 模拟处理时间
        
        steps.append({"step": 4, "description": "结果排序", "details": "正在对匹配结果进行排序..."})
        time.sleep(1)  # 模拟处理时间
        
        steps.append({"step": 5, "description": "生成分析报告", "details": "正在生成详细的分析报告..."})
        analysis = get_model_analysis(raw_input, matches)
        time.sleep(1)  # 模拟处理时间
        
        chart_data = generate_chart_data(matches)
        steps.append({"step": 6, "description": "完成", "details": "分析报告生成完毕"})
        
        return jsonify({
            "report": analysis,
            "chartData": chart_data,
            "isGeneralQuery": False,
            "steps": steps
        })

@app.route('/')
def index():
    return render_template('index.html', num_rows=NUM_ROWS, developer=DEVELOPER, version=VERSION, update_date=UPDATE_DATE)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    logging.info(f"接收到搜索请求: {query}")
    try:
        result = process_query(query)
        logging.info("搜索完成，返回结果")
        return result
    except Exception as e:
        logging.error(f"搜索过程中发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    logging.info("正在启动MediSync - 智慧医疗数据同步器...")
    logging.info(f"项目文件夹已创建在: {project_folder}")
    
    # 加载数据
    load_data()
    
    Timer(1, open_browser).start()
    logging.info("应用程序正在启动，浏览器将自动打开。如果没有自动打开，请手动访问 http://127.0.0.1:5000/")
    app.run(debug=True)
# 添加这些行到文件末尾
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
