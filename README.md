![Python Badge](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))  
![Author Badge](https://img.shields.io/badge/Author-Benoit_Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2023--10--02-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

# ETH-ScopeM_Arora
Phase-separated liquid droplets classification 
## Description

## Installation
In this tutorial, we will see how to install [Python](https://www.python.org/) using a [Conda](https://docs.conda.io/en/latest/) package manager to execute our scripts in a controlled environment with all essential dependencies.  

### Download GitHub repository:  

- Download this repository by clicking the following 
[link](https://github.com/BDehapiot/ETH-ScopeM_Arora/archive/refs/heads/main.zip)  
- Unzip the downloaded folder to a known location (e.g. `Desktop`)

### Install Conda and create a new environment:

We will now install Conda using the light-weight [Miniforge](https://github.com/conda-forge/miniforge) installer and create a new environment using the `environment.yml` file shipped with this repository.

Select your operating system:  

<details> <summary>Windows</summary>  

- Download Miniforge installer for Windows
([link](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe))  

- Run the downloaded `.exe` file and select the following options:    
    - "create start menu shortcuts"  
    - "add Miniforge3 to PATH environment variable" 

- Run Miniforge Prompt from your start menu shortcuts  

    Your prompt should read something like:  
    `(base) C:\Users\YourUsername>`  
    `(base)` meaning that you are in your base Conda environment 

- Move to your downloaded GitHub repository using the `cd` command: 
    ```bash
    cd Desktop/ETH-ScopeM_Arora-main
    ```
    Your prompt should change to reflect your current location:  
    `(base) C:\Users\YourUsername\Desktop\ETH-ScopeM_Arora-main>`

- Create a new Conda environment (takes a few minutes): 
    ```bash
    mamba env create -f environment.yml
    ```

- Activate the new environment:
    ```bash
    conda activate Arora
    ```
    Your prompt should now display `(Arora)` indicating that you have changed environment   
    `(Arora) C:\Users\YourUsername\Desktop\ETH-ScopeM_Arora-main>`

</details> 

<details> <summary>MacOS</summary>  

- Download Miniforge installer for MacOS 
([Intel-Series](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh))
([M-Series](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)) 

- Open your terminal by typing `terminal` in the Launchpad  

    Your prompt should read something like:  
    `YourUsername@MacBook-Pro ~ %`

- Move to where you downloaded the Miniforge installer using the `cd` command:  
It is most likely located in your `Downloads` folder    
    ```bash
    cd ~/Downloads
    ```  
    
- Run the following command to install Miniforge:  

    ```bash
    # Intel-Series
    bash Miniforge3-MacOSX-x86_64.sh
    # M-Series
    bash Miniforge3-MacOSX-arm64.sh
    ```  
    Follow the Terminal prompts to complete installation and accept default options  

- Close and re-open your terminal  

    Your prompt should now read something like:  
    `(base) YourUsername@MacBook-Pro ~ %`  
    `(base)` meaning that you are in your base Conda environment  

- Move to your downloaded GitHub repository: 
    ```bash
    cd Desktop/ETH-ScopeM_Arora-main
    ```
    Your prompt should change to reflect your current location:  
    `(base) YourUsername@MacBook-Pro Desktop/ETH-ScopeM_Arora-main %`  

- Create a new Conda environment (takes a few minutes):  
    ```bash
    mamba env create -f environment.yml
    ```

- Activate the new environment:
    ```bash
    conda activate Arora
    ```

    Your prompt should now display `(Arora)` indicating that you have changed environment  
    `(Arora) YourUsername@MacBook-Pro Desktop/ETH-ScopeM_Arora-main %`

</details> 

### Execute Python scripts: 

To execute Python scripts, identifiable by their `.py` extension, you can either enter commands in your terminal or use an Integrated Development Environment (IDE) to directly interact with the code (preferred option). Here, we'll guide you on how to install and set up the [Spyder](https://docs.spyder-ide.org/current/index.html) IDE, which is a favorable choice for beginners.

- Open your terminal and activate your new environment:  
    ```bash
    conda activate Arora
    ```

- Install Spyder using pip (only the first time):
    ```bash
    pip install spyder
    ```

- Launch spyder and wait for the graphical interface to appear:
    ```bash
    spyder
    ```

- Use the interface to create a new project:
    - Click `Projects` > `New Project...`
    - Choose `Existing directory`
    - Select the downloaded GitHub repository using the folder icon
    - Click the `Create` button  

    Projects can be re-opened later with: `Projects` > `Recent Projects...`

You can now browse the repository to open, modify and execute `.py` Python scripts within Spyder.

## Dependencies
### Conda
- python=3.10
- numpy
- scipy
- scikit-image
- pandas
- opencv
- joblib
- matplotlib-base
- pyyaml
- scikit-learn
- nd2
- pip

### pip
- napari[all]
