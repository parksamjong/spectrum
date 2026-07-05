import os
import sys
import threading
import time
import webbrowser

# PyInstaller 번들 실행 시 작업 디렉터리를 exe 위치로 설정
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)

# .env를 exe 옆에서 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

def _open_browser():
    import socket, subprocess
    # 서버가 실제로 응답할 때까지 최대 30초 대기
    for _ in range(30):
        try:
            s = socket.create_connection(('localhost', 8023), timeout=1)
            s.close()
            break
        except OSError:
            time.sleep(1)
    # Windows: start 명령으로 기본 브라우저 오픈 (webbrowser 모듈 폴백)
    try:
        subprocess.Popen(['cmd', '/c', 'start', '', 'http://localhost:8023'],
                         shell=False, creationflags=0x08000000)
    except Exception:
        webbrowser.open('http://localhost:8000')

if __name__ == '__main__':
    threading.Thread(target=_open_browser, daemon=True).start()
    import uvicorn
    from main import app
    uvicorn.run(app, host='0.0.0.0', port=8023)
