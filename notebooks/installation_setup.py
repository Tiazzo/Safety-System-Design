import os
import subprocess
import sys

# Specify the virtual environment name
venv_name = "name_venv" # Change this to your desired name

# Path to the virtual environment
venv_path = os.path.join(os.getcwd(), venv_name)

# Path to the .gitignore file
gitignore_path = os.path.join(os.getcwd(), ".gitignore")

# Step 1: Create the virtual environment if it doesn't exist
if not os.path.exists(venv_path):
    print(f"Creating virtual environment '{venv_name}'...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
else:
    print(f"Virtual environment '{venv_name}' already exists.")

# Step 2: Add virtual environment to .gitignore
if os.path.exists(gitignore_path):
    with open(gitignore_path, "r") as gitignore_file:
        gitignore_content = gitignore_file.readlines()

    # Check if the venv path is already in .gitignore
    if venv_name + "/" not in [line.strip() for line in gitignore_content]:
        with open(gitignore_path, "a") as gitignore_file:
            gitignore_file.write(f"\n{venv_name}/\n")
        print(f"Added '{venv_name}/' to .gitignore.")
    else:
        print(f"'{venv_name}/' is already in .gitignore.")
else:
    # Create .gitignore and add the virtual environment
    with open(gitignore_path, "w") as gitignore_file:
        gitignore_file.write(f"{venv_name}/\n")
    print(f".gitignore created and added '{venv_name}/'.")

# Step 3: Install required libraries
print("Installing required libraries in the virtual environment...")
activate_script = (
    f'"{os.path.join(venv_path, "Scripts", "activate")}"'  # Windows
    if os.name == "nt"
    else f'"{os.path.join(venv_path, "bin", "activate")}"'  # Linux/Mac
)
pip_install_cmd = f"source {activate_script} && pip install jupyter ipykernel opencv-python numpy matplotlib torch torchvision pillow"

# Use subprocess to execute the command
subprocess.run(pip_install_cmd, shell=True, check=True)

# Step 4: Register the virtual environment as a Jupyter kernel
print("Registering the virtual environment as a Jupyter kernel...")
register_kernel_cmd = f"source {activate_script} && python -m ipykernel install --user --name={venv_name} --display-name 'Python ({venv_name})'"
subprocess.run(register_kernel_cmd, shell=True, check=True)

print(f"Kernel 'Python ({venv_name})' is now available in Jupyter.")
print("Select the kernel to use the virtual environment in Jupyter notebooks.")