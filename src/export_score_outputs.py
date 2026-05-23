"""
Giữ cho score_outputs.csv đồng bộ tuyệt đối với run_all_models.py.

Thay vì có một pipeline riêng dễ lệch logic,
file này gọi trực tiếp run_experiment() từ src.run_all_models.
"""

from src.run_all_models import run_experiment


def main():
    _, detailed_df = run_experiment()
    print(f"\nĐã tạo score_outputs.csv với {len(detailed_df)} dòng.")


if __name__ == "__main__":
    main()