import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    from data_mapping import FINAL_LABEL_SET, map_chexpert_labels
except ImportError:
    DISEASES = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 
        'Pleural Effusion', 'Pneumothorax'
    ]
    FINAL_LABEL_SET = ['No Finding'] + DISEASES

    def map_chexpert_labels(df_raw):
        df_mapped = df_raw[['Path'] + FINAL_LABEL_SET].copy()
        df_mapped = df_mapped.fillna(0)
        for col in FINAL_LABEL_SET:
            if col in df_mapped.columns:
                df_mapped[col] = df_mapped[col].replace(-1.0, 1.0)
        return df_mapped

CHEXPERT_PATH = "./datasets/CheXpert-v1.0-small" # Đường dẫn đến bộ dữ liệu gốc
TRAIN_CSV_FILENAME = "train.csv"
OUTPUT_DIR = "./analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_chexpert_imbalance():
    """
    Phân tích và trực quan hóa sự mất cân bằng lớp trong tập train của CheXpert.
    """
    print("--- Starting CheXpert Data Imbalance Analysis ---")
    
    # Tải và xử lý dữ liệu
    csv_path = os.path.join(CHEXPERT_PATH, TRAIN_CSV_FILENAME)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print(f"Loading and mapping data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    df = map_chexpert_labels(df_raw)
    
    label_df = df[FINAL_LABEL_SET]
    total_samples = len(label_df)

    # Tính toán tần suất và tỷ lệ phần trăm
    print("\n--- Class Frequency and Percentage ---")
    class_counts = label_df.sum()
    class_percentages = (class_counts / total_samples) * 100
    
    # Tạo một DataFrame để hiển thị đẹp hơn
    summary_df = pd.DataFrame({
        'Count': class_counts,
        'Percentage (%)': class_percentages.round(2)
    }).sort_values(by='Count', ascending=False)
    
    print(summary_df)

    # Trực quan hóa tần suất lớp (ĐÃ CẬP NHẬT)
    plt.figure(figsize=(14, 8))
    # Sử dụng dữ liệu đã sắp xếp từ summary_df
    ax = sns.barplot(x=summary_df.index, y=summary_df['Count'], palette="viridis")
    
    plt.title('Class Distribution in CheXpert Training Set', fontsize=18, pad=20)
    plt.ylabel('Number of Positive Samples', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    for i, p in enumerate(ax.patches):
        # Lấy chiều cao của cột (số lượng)
        count = int(p.get_height())
        # Lấy tỷ lệ phần trăm từ summary_df
        percentage = summary_df['Percentage (%)'][i]
        
        # Tạo chuỗi văn bản để hiển thị
        label_text = f'{count}\n({percentage:.1f}%)'
        
        # Vị trí để đặt văn bản
        x = p.get_x() + p.get_width() / 2.
        y = p.get_height()
        
        # Thêm văn bản vào biểu đồ
        ax.annotate(label_text, (x, y), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=11, color='black')

    # Lưu biểu đồ
    save_path_bar = os.path.join(OUTPUT_DIR, "chexpert_class_distribution_detailed.png")
    plt.savefig(save_path_bar)
    print(f"\nClass distribution bar chart saved to: {save_path_bar}")
    plt.show()

    # Phân tích ma trận đồng xuất hiện (giữ nguyên)
    disease_df = label_df.drop(columns=['No Finding'])
    diseases_list = disease_df.columns.tolist()
    
    co_occurrence_matrix = pd.DataFrame(index=diseases_list, columns=diseases_list, dtype=int)
    for disease1 in diseases_list:
        for disease2 in diseases_list:
            count = len(disease_df[(disease_df[disease1] == 1) & (disease_df[disease2] == 1)])
            co_occurrence_matrix.loc[disease1, disease2] = count

    print("\n--- Disease Co-occurrence Matrix ---")
    print(co_occurrence_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title('Disease Co-occurrence Heatmap in CheXpert', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path_heatmap = os.path.join(OUTPUT_DIR, "chexpert_co_occurrence_heatmap.png")
    plt.savefig(save_path_heatmap)
    print(f"\nCo-occurrence heatmap saved to: {save_path_heatmap}")
    plt.show()

    print("\n--- Analysis Completed ---")


if __name__ == "__main__":
    analyze_chexpert_imbalance()