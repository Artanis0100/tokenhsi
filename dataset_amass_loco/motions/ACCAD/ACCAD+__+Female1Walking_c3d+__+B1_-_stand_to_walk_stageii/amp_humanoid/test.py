import numpy as np
import os

def read_first_1000(filename):
    # Check if file exists in the current directory
    if not os.path.exists(filename):
        print(f"❌ Error: The file '{filename}' was not found in the current directory.")
        print(f"Current directory is: {os.getcwd()}")
        return

    try:
        # 1. Load the entire array
        # mmap_mode='r' is useful here: it reads the file without loading 
        # the whole thing into RAM, which is great if your file is huge.
        data = np.load(filename, mmap_mode='r')
        
        print(f"✅ File loaded successfully.")
        print(f"Original Shape: {data.shape}")
        print(f"Original Type:  {data.dtype}")

        # 2. Flatten the data to 1D
        # This ensures we get the first 1000 'numbers' regardless of if 
        # the original shape is (N, 3), (H, W, C), etc.
        flat_data = data.flatten()

        # 3. Get the first 1000 elements
        # Python slicing is safe; if there are < 1000 elements, it takes all of them.
        first_1000 = flat_data[:1000]

        # 4. Output results
        print("-" * 30)
        print(f"Preview of first {len(first_1000)} numbers:\n")
        print(first_1000)
        print("-" * 30)
        
        # Optional: Save to text file if you need to copy them
        # np.savetxt("output_1000.txt", first_1000)

    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")

# ==========================================
# CHANGE THIS VARIABLE TO YOUR FILE NAME
# ==========================================
target_file = "ref_motion.npy" 

if __name__ == "__main__":
    # If you didn't change the name above, let's try to auto-find a .npy file
    if target_file == "my_data.npy" and not os.path.exists(target_file):
        npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]
        if npy_files:
            target_file = npy_files[0]
            print(f"⚠️ 'my_data.npy' not found. Auto-detected '{target_file}' instead.\n")
    
    read_first_1000(target_file)
