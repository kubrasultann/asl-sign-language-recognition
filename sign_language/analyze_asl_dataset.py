import pandas as pd

DATASET_PATH = "sign_language/dataset_asl.csv"

def main():
    print("=== ASL Dataset Analizi ===\n")

    try:
        df = pd.read_csv(DATASET_PATH, header=None)
    except FileNotFoundError:
        print("❌ dataset_asl.csv bulunamadı!")
        return

    # Son sütun label
    labels = df.iloc[:, -1]

    print(f"Toplam satır sayısı: {len(df)}\n")

    print("Sınıf sayıları:")
    counts = labels.value_counts()
    print(counts)
    print()

    print("Sınıf yüzdeleri (%):")
    percentages = (counts / counts.sum()) * 100
    print(percentages.round(2))
    print()

    # Kontroller
    print("Boş label var mı?:", labels.isna().any())
    print("Boş satır var mı?:", df.isna().any().any())

    print("\n=== Özet ===")
    print(f"Toplam sınıf sayısı: {labels.nunique()}")
    print(f"En az örneği olan sınıf: {counts.idxmin()} ({counts.min()})")
    print(f"En çok örneği olan sınıf: {counts.idxmax()} ({counts.max()})")

if __name__ == "__main__":
    main()
