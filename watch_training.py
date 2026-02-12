"""
实时监控训练进度
"""
import time
import os

output_file = r"D:\dell\temp\claude\F--Claude-code-EVRL260207\tasks\b07b256.output"

print("="*60)
print("训练进度监控 (按 Ctrl+C 退出)")
print("="*60)
print()

last_size = 0
while True:
    try:
        if os.path.exists(output_file):
            current_size = os.path.getsize(output_file)
            if current_size > last_size:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    print(new_content, end='')
                last_size = current_size
        time.sleep(2)
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
