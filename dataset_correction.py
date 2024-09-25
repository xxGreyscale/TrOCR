import pandas as pd
import argparse


def correct_paths(input_file, output_file, prefix):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Replace incorrect path separators
    df['image_path'] = df['image_path'].str.replace(r'\\+', '/', regex=True)
    df['image_path'] = df['image_path'].str.replace(r'\\', '/', regex=True)   # Handles \

    # remove everything except the last part of the path, and add images/ to the beginning
    # if prefix:
    #     df['image_path'] = prefix + df['image_path'].str.split('/').str[-1]
    # df['image_path'] = df['image_path'].str.split('/').str[-1]
    # df['image_path'] = 'images/' + df['image_path']

    #  For one time use, tak the last 2 parts of the path
    df['image_path'] = prefix + df['image_path'].str.split('/').str[-2] + '/' + df['image_path'].str.split('/').str[-1]

    # Save the corrected DataFrame back to the CSV
    df.to_csv(output_file, index=False)

    print(f"Path separators corrected and saved to '{output_file}'.")


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Correct path separators in a CSV file.')

    # Add arguments for input and output file paths
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_file', type=str, help='Path to save the corrected CSV file.')
    parser.add_argument('prefix', type=str, help='Prefix to add to the corrected path. Example if we want the directory to be images/')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to correct paths
    correct_paths(args.input_file, args.output_file, args.prefix)


if __name__ == "__main__":
    main()
