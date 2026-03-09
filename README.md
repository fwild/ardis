# Augmented Reality Discovery (ardis)
Fridolin Wild (f.wild@open.ac.uk)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

This is a foundational example for multimodal AR discovery using 
the You-Only-Look-Once (YOLO) computer vison models.

# Installation

## 1. Clone the Repository

``` bash
git clone https://github.com/fwild/ardis.git 
cd ardis
```

## 2. Create a Python Virtual Environment

``` bash
python -m venv .venv
```

## 3. Activate the Environment

### macOS / Linux

``` bash
source .venv/bin/activate
```

### Windows (PowerShell)

``` powershell
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

``` cmd
.venv\Scripts\activate
```

## 4. Install Dependencies

``` bash
pip install -r requirements.txt
```

# Environment Variables

This project uses environment variables for configuration.

Create a `env.sh` file in the project root:

```
export ROBOFLOW_API_KEY=your_api_key_here
```

# Running the Project

Run the main application:

``` bash
./env.sh
python main.py
```

