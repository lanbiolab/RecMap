import ast
import pandas as pd
import os
import json


def process_json_file(json_path):
    print(f"Processing JSON files: {json_path}")
    products = []

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            full_data = json.load(f)
            if isinstance(full_data, list):
                print(f"A complete JSON array was detected, containing {len(full_data)} records.")
                products.extend(full_data)
                return products
            elif isinstance(full_data, dict):
                print("A single JSON object was detected.")
                products.append(full_data)
                return products
        except json.JSONDecodeError:
            print("Unable to parse as a complete JSON file, attempting to process...")
            f.seek(0)

        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            count += 1
            try:
                data = json.loads(line)
                products.append(data)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(line)
                    products.append(data)
                except Exception:
                    print(f"Lines that failed to be parsed: {count}: {line[:100]}")
                    continue

        print(f"The {count} rows were processed, and {len(products)} objects were successfully parsed.")

    return products


def build_product_dict(products):
    product_dict = {}
    for product in products:
        asin = product.get('asin')
        if asin:
            asin = str(asin).strip()
            product_dict[asin] = process_product(product)

    print(f"{len(product_dict)} valid product entries were created.")
    return product_dict


def process_product(product):
    imUrl = product.get('imUrl', 'none')

    title = product.get('title', 'none')
    price = product.get('price')
    brand = product.get('brand', 'none')

    price_str = "none" if price is None else f"{price}"

    categories = product.get('categories', [])
    if categories and isinstance(categories, list):
        flat_categories = []
        for cat in categories:
            if isinstance(cat, list):
                flat_categories.extend([str(c).strip() for c in cat])
            else:
                flat_categories.append(str(cat).strip())
        categories_str = ", ".join(flat_categories)
    else:
        categories_str = "none"

    description = product.get('description', 'none')

    text = (
        f"The product title is {title}. "
        f"The product price is {price_str}. "
        f"The product brand is {brand}. "
        f"The product categories are {categories_str}. "
        f"The product description is {description}."
    )

    return {
        'imUrl': imUrl,
        'text': text
    }


def process_csv_mapping(csv_path):
    print(f"Processing CSV files: {csv_path}")

    sample_lines = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for _ in range(5):
            sample_lines.append(f.readline().strip())

    print("CSV sample line:")
    for i, line in enumerate(sample_lines):
        print(f"{i}: {line}")

    is_tsv = any('\t' in line for line in sample_lines)
    has_header = any('asin' in line.lower() for line in sample_lines)

    sep = '\t' if is_tsv else ','
    header = 0 if has_header else None
    names = ['asin', 'itemID'] if not has_header else None

    df = pd.read_csv(
        csv_path,
        sep=sep,
        header=header,
        names=['asin', 'itemID'],
        dtype=str,
        on_bad_lines='warn'
    )

    print(f"Successfully read {len(df)} mapping records.")
    print(df.head())

    if 'itemID' not in df.columns:
        if len(df.columns) == 2:
            df.columns = ['asin', 'itemID']
        elif 'asin' in df.columns:
            df['itemID'] = df.index.astype(str)
        else:
            raise ValueError(f"Unrecognized CSV format: {df.columns.tolist()}")

    return df


def main():
    data_dir = r'.\Data\clothing'
    json_path = os.path.join(data_dir, 'meta_Clothing_Shoes_and_Jewelry.json')
    csv_path = os.path.join(data_dir, 'i_id_mapping.csv')
    output_path = os.path.join(data_dir, 'combined_output.csv')

    print("=" * 50)
    print("Start processing...")

    products = process_json_file(json_path)
    product_dict = build_product_dict(products)

    df_mapping = process_csv_mapping(csv_path)

    print("\nMerging data...")
    results = []
    asin_in_csv = set()
    asin_in_json = set()

    for _, row in df_mapping.iterrows():
        asin = str(row['asin']).strip()
        item_id = str(row['itemID']).strip()

        asin_in_csv.add(asin)

        product_info = product_dict.get(asin)
        if product_info:
            asin_in_json.add(asin)
            results.append({
                'asin': asin,
                'itemID': item_id,
                'imUrl': product_info['imUrl'],
                'text': product_info['text']
            })
        else:
            results.append({
                'asin': asin,
                'itemID': item_id,
                'imUrl': 'none',
                'text': (
                    "The product title is none. The product price is none. "
                    "The product brand is none. The product categories are none. "
                    "The product description is none."
                )
            })

    print(f"Quantity of asin in CSV: {len(asin_in_csv)}")
    print(f"Number of matches found in JSON: {len(asin_in_json)}")

    df_output = pd.DataFrame(results)

    df_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nThe output has been saved to: {output_path}")

    print("\nOutput the first 5 lines:")
    print(df_output.head())

    missing_text = df_output[df_output['text'].str.contains("title is none")]
    print(f"\nNumber of records where product information could not be found: {len(missing_text)}")
    found_text = df_output[~df_output['text'].str.contains("title is none")]
    print(f"Number of records containing product information: {len(found_text)}")

    if len(found_text) > 0:
        print("\nExamples of products found:")
        print(found_text.sample(min(3, len(found_text))))


if __name__ == '__main__':
    main()