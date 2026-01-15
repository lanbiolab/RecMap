import numpy as np
import os


def filter_interactions(npy_path, inter_path, output_npy_path=None):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"{npy_path} not found")

    data = np.load(npy_path, allow_pickle=True).item()
    print(f"Loading the file successfully, containing {len(data)} users.")

    user_item_xlabel = {}
    total_inters = 0
    with open(inter_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            user_id = parts[0]
            item_id = parts[1]

            try:
                x_label = int(parts[4])
            except ValueError:
                continue

            key = (str(user_id), str(item_id))
            user_item_xlabel[key] = x_label
            total_inters += 1

    print(f"Load {total_inters} interaction records from the .inter file.")

    total_items = 0
    kept_count = 0
    removed_count = 0
    missing_count = 0

    for user_id, (interactions, extra) in data.items():
        new_interactions = []

        str_user_id = str(user_id)

        for item_id in interactions:
            total_items += 1
            str_item_id = str(item_id)
            key = (str_user_id, str_item_id)

            if key in user_item_xlabel:
                if user_item_xlabel[key] == 0:
                    new_interactions.append(item_id)
                    kept_count += 1
                else:
                    removed_count += 1
            else:
                new_interactions.append(item_id)
                missing_count += 1

        data[user_id] = [new_interactions, extra]

    print("\n" + "=" * 50)
    print(f"Processing statistics:")
    print(f"  Total number of items: {total_items}")
    print(f"  Number of retained items (x_label=0): {kept_count}")
    print(f"  Number of deleted items (x_label=1 or 2): {removed_count}")
    print(f"  No interaction history found, but number of items retained: {missing_count}")
    print("=" * 50)

    output_path = output_npy_path or npy_path
    np.save(output_path, data)
    print(f"\nThe results have been saved to: {output_path}")


if __name__ == "__main__":
    filter_interactions(
        npy_path=r".\Data\clothing\clothing_user_item_matrix.npy",
        inter_path=r".\Data\clothing\clothing.inter",
        output_npy_path=r".\Data\clothing\new_clothing_user_item_matrix.npy"
    )