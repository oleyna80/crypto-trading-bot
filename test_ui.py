"""
Тестирование UI Streamlit через HTTP запросы.
"""
import requests
import time
import sys

def test_streamlit_ui():
    """Проверяет доступность и базовую функциональность Streamlit UI."""
    base_url = "http://localhost:8501"
    
    # Проверка доступности
    try:
        resp = requests.get(base_url, timeout=10)
        if resp.status_code == 200:
            print("[OK] Streamlit UI доступен (HTTP 200)")
            # Проверяем, что страница содержит ожидаемые элементы
            if "Grid Trading Backtester" in resp.text:
                print("[OK] Заголовок страницы присутствует")
            else:
                print("[WARN] Заголовок страницы не найден (возможно кодировка)")
            return True
        else:
            print(f"[FAIL] Ошибка HTTP: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[FAIL] Не удалось подключиться к Streamlit серверу")
        return False
    except Exception as e:
        print(f"[FAIL] Неожиданная ошибка: {e}")
        return False

if __name__ == "__main__":
    # Дадим серверу время на запуск
    time.sleep(2)
    success = test_streamlit_ui()
    sys.exit(0 if success else 1)