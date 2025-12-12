import matplotlib.pyplot as plt
import re
import numpy as np
import os

# --- CẤU HÌNH ---
# Hãy đổi tên này trùng khớp hoàn toàn với tên file log của bạn
LOG_FILE = 'training_log20_10.txt'  
# Độ mượt của biểu đồ (số càng to càng mượt, dễ nhìn xu hướng)
WINDOW_SIZE = 50  

def parse_log(filename):
    if not os.path.exists(filename):
        # Thử thêm đuôi .txt nếu người dùng quên
        if os.path.exists(filename + ".txt"):
            filename += ".txt"
        else:
            print(f"❌ LỖI: Không tìm thấy file '{filename}'")
            return [], [], [], []

    print(f"Đang đọc file: {filename}...")
    
    episodes = []
    rewards = []
    
    vali_episodes = []
    vali_makespans = []

    current_ep = 0

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1. Bắt dòng Training
            # Mẫu: Ep   10 | Reward: -1064.50 | ...
            train_match = re.search(r'Ep\s+(\d+)\s+\|\s+Reward:\s+(-?\d+\.\d+)', line)
            if train_match:
                current_ep = int(train_match.group(1))
                episodes.append(current_ep)
                rewards.append(float(train_match.group(2)))

            # 2. Bắt dòng Validation
            # Mẫu: >>> VALIDATION: 1851.49 ...
            val_match = re.search(r'VALIDATION:\s+(\d+\.\d+)', line)
            if val_match:
                makespan = float(val_match.group(1))
                # Gán kết quả validation này cho episode vừa chạy xong
                if current_ep > 0:
                    vali_episodes.append(current_ep)
                    vali_makespans.append(makespan)

    return episodes, rewards, vali_episodes, vali_makespans

def moving_average(data, window_size):
    """Hàm làm mượt biểu đồ"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(ep, rw, v_ep, v_mk):
    # Tạo 2 biểu đồ trên dưới
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- BIỂU ĐỒ 1: REWARD (Training) ---
    # Vẽ mờ dữ liệu gốc (nhiễu)
    ax1.plot(ep, rw, alpha=0.2, color='gray', label='Raw Reward')
    
    # Vẽ đậm dữ liệu đã làm mượt (xu hướng chính)
    if len(rw) > WINDOW_SIZE:
        smooth_rw = moving_average(rw, WINDOW_SIZE)
        smooth_ep = ep[len(ep)-len(smooth_rw):]
        ax1.plot(smooth_ep, smooth_rw, color='blue', linewidth=2, label=f'Trend (MA-{WINDOW_SIZE})')
    
    ax1.set_ylabel('Reward (Càng cao càng tốt)', fontsize=12)
    ax1.set_title('Training Progress: Reward over 10,000 Episodes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- BIỂU ĐỒ 2: MAKESPAN (Validation) ---
    if len(v_mk) > 0:
        ax2.plot(v_ep, v_mk, color='red', linewidth=2, label='Validation Makespan')
        
        # Tìm điểm thấp nhất (Best Model)
        min_idx = np.argmin(v_mk)
        min_val = v_mk[min_idx]
        min_ep = v_ep[min_idx]
        
        # Đánh dấu điểm tốt nhất
        ax2.scatter(min_ep, min_val, color='gold', s=150, edgecolors='black', zorder=5, label='Best Model')
        ax2.annotate(f'Best: {min_val:.1f}\n(Ep {min_ep})', 
                     xy=(min_ep, min_val), 
                     xytext=(min_ep, min_val + (max(v_mk)-min_val)*0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, fontweight='bold', ha='center')

        ax2.set_ylabel('Makespan (Càng thấp càng tốt)', fontsize=12)
        ax2.set_title('Validation Performance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "Không tìm thấy dữ liệu Validation trong file log", ha='center', fontsize=12)

    plt.xlabel('Episodes', fontsize=12)
    plt.tight_layout()
    
    # Lưu ảnh trước khi show
    save_path = 'result_plot_10000eps.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ Đã lưu biểu đồ vào file: {save_path}")
    plt.show()

if __name__ == '__main__':
    episodes, rewards, v_episodes, v_makespans = parse_log(LOG_FILE)
    
    if len(episodes) > 0:
        print(f"Đã đọc được {len(episodes)} dòng dữ liệu training.")
        print(f"Đã đọc được {len(v_makespans)} điểm validation.")
        plot_results(episodes, rewards, v_episodes, v_makespans)
    else:
        print("Không đọc được dữ liệu nào. Hãy kiểm tra lại tên file log.")