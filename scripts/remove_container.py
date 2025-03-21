import docker
from tqdm import tqdm


def remove_stopped_containers():
    """
    Remove all Docker containers in 'exited' or 'created' states.
    """
    client = docker.from_env()  # Connect to the Docker daemon
    try:
        # Get all containers, including stopped and created ones
        containers = client.containers.list(all=True)
        
        # Filter containers that are either 'exited' or 'created'
        removable_containers = [
            container for container in containers if container.status in ["exited", "created"]
        ]
        
        if not removable_containers:
            print("No containers in 'exited' or 'created' state found.")
            return

        print(f"Found {len(removable_containers)} containers in 'exited' or 'created' state. Removing them...")

        # Use tqdm for progress bar
        for container in tqdm(removable_containers, desc="Removing containers"):
            try:
                container.remove()
                print(f"[SUCCESS] Removed container: {container.name} (ID: {container.id}) - Status: {container.status}")
            except Exception as e:
                print(f"[FAILED] Could not remove container: {container.name} (ID: {container.id}) - Error: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Docker: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    remove_stopped_containers()