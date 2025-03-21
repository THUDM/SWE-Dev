import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def get_all_subfolders(folder_path):
    """
    Recursively gather all subfolder paths under the given folder using os.listdir.
    """
    useful = ["swedev", "swedev", "base"]
    return [f'{folder_path}/{path}' for path in os.listdir(folder_path) if not path in useful]

def delete_folder(folder_path):
    """
    Delete a single folder and its contents.
    """
    try:
        shutil.rmtree(folder_path)  # Recursively delete the folder and its contents
        return folder_path, True, None
    except Exception as e:
        return folder_path, False, str(e)

def concurrent_delete(parent_folder, max_workers=8):
    """
    Concurrently delete all subfolders under the parent folder.
    """
    # Get all subfolders using a recursive listdir approach
    subfolders = get_all_subfolders(parent_folder)

    # Sort subfolders by depth, ensuring the deepest folders are deleted first
    subfolders.sort(key=lambda x: x.count(os.sep), reverse=True)

    # Add the parent folder itself to the list of folders to delete
    subfolders.append(parent_folder)

    # Use a thread pool to delete folders concurrently
    print(f"Starting deletion of {len(subfolders)} folders...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(delete_folder, folder): folder for folder in subfolders}

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_folder), total=len(future_to_folder), desc="Deleting folders"):
            folder_path, success, error = future.result()
            if success:
                print(f"[SUCCESS] Deleted: {folder_path}")
            else:
                print(f"[FAILED] Could not delete: {folder_path} - Error: {error}")

if __name__ == "__main__":
    target_folder = "/raid/playground"
    if os.path.exists(target_folder):
        concurrent_delete(target_folder, max_workers=64)
    else:
        print(f"[ERROR] Path does not exist: {target_folder}")
