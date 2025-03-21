import os

import matplotlib.pyplot as plt

def main():
    INPUT_FILE = "results/issues/all_tasks_with_test.jsonl"
    OUTPUT_FILE = "filtered_test.jsonl"

    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    line_sizes = []  # Stores the size of each line
    filtered_lines = []  # Stores lines <= 1MB

    # Read the file and process line sizes
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            size = len(line.encode("utf-8"))  # Calculate size in bytes
            line_sizes.append(size)
            if size <= 1024 * 1024:  # Filter lines <= 1MB
                filtered_lines.append(line)

    # Save filtered lines to a new file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

    print(f"Total lines: {len(line_sizes)}")
    print(f"Lines <= 1MB: {len(filtered_lines)}")
    print(f"Filtered lines saved to '{OUTPUT_FILE}'.")

    # If no lines are <= 1MB, stop further processing
    if not filtered_lines:
        print("No lines <= 1MB found.")
        return

    # Convert the sizes of filtered lines to KB for visualization
    filtered_sizes_kb = [len(line.encode("utf-8")) / 1024 for line in filtered_lines]

    # Plot the size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        filtered_sizes_kb,
        bins=50,
        color="blue",
        edgecolor="black",
        alpha=0.7
    )
    plt.title("Line Size Distribution (Filtered, <= 1MB)", fontsize=15)
    plt.xlabel("Line Size (KB)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig("line_size_distribution.png")
    print("Distribution chart saved as 'line_size_distribution.png'.")
    plt.show()

if __name__ == "__main__":
    main()