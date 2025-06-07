import re
import csv
import os

def parse_rl_log(log_file_path, output_csv_path):
    """
    Parse the reinforcement learning log file and extract episode data into a CSV file
    with episode number, reward, key_picked, door_opened, goal_reached, and steps.
    
    Args:
        log_file_path (str): Path to the log file
        output_csv_path (str): Path to save the output CSV file
    """
    # Read the log file
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Define regex pattern to extract episode information
    pattern = r'Episode (\d+)/\d+ \| Reward: ([\-\d\.]+) \| Steps: (\d+) \| Key: (✅|❌), Door: (✅|❌), Goal: (✅|❌)'
    
    # Find all matches
    matches = re.findall(pattern, log_content)
    
    # Prepare data for CSV
    episodes_data = []
    for match in matches:
        episode = int(match[0])
        reward = float(match[1])
        steps = int(match[2])
        key_picked = True if match[3] == '✅' else False
        door_opened = True if match[4] == '✅' else False
        goal_reached = True if match[5] == '✅' else False
        
        episodes_data.append({
            'episode': episode,
            'reward': reward,
            'length': steps,
            'key_picked': key_picked,
            'door_opened': door_opened,
            'goal_reached': goal_reached
        })
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'reward', 'length', 'key_picked', 'door_opened', 'goal_reached']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in episodes_data:
            writer.writerow(data)
    
    print(f"Successfully extracted data from {len(episodes_data)} episodes")
    print(f"CSV file saved to {output_csv_path}")

if __name__ == "__main__":
    # Set your input and output file paths
    input_file = "op.txt"  # Path to your log file
    output_file = "episode_metrics_4x4.csv"  # Path to save the CSV
    
    parse_rl_log(input_file, output_file)