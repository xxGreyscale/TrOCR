import pandas as pd
import argparse


def correct_paths(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Replace incorrect path separators
    df['image_path'] = df['image_path'].str.replace(r'\\+', '/', regex=True)
    df['image_path'] = df['image_path'].str.replace(r'\\', '/', regex=True)   # Handles \

    # Save the corrected DataFrame back to the CSV
    df.to_csv(output_file, index=False)

    print(f"Path separators corrected and saved to '{output_file}'.")


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Correct path separators in a CSV file.')

    # Add arguments for input and output file paths
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_file', type=str, help='Path to save the corrected CSV file.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to correct paths
    correct_paths(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
