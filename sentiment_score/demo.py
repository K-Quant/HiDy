import argparse

from sentiment_sore import extract_sentiment_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_report_path", type=str, default=None,
                        help="The path of input text data that you want to load.")
    parser.add_argument("--extracted_data_path", type=str, default=None,
                        help="The path of extracted knowledge that you want to save in.")
    args = parser.parse_args()

    extract_sentiment_score(args.original_report_path, args.extracted_data_path)