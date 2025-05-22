import threading
from base_ctrl import BaseController

base = BaseController('/dev/ttyUSB0', 115200)

L = 0.0
R = 0.0

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def main():
    global L, R
    print("실시간 제어: '왼쪽값 오른쪽값' 입력 후 엔터 (예: 0.2 0.2), 종료는 Ctrl+C")
    while True:
        try:
            inp = input(f"현재 L={L:.2f}, R={R:.2f} > ")
            parts = inp.strip().split()
            if len(parts) != 2:
                print("입력 예시: 0.2 0.2")
                continue
            L, R = float(parts[0]), float(parts[1])
            send_control_async(L, R)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")

if __name__ == "__main__":
    main()