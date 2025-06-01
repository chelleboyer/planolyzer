# Planolyzer

A planogram analysis tool that helps verify if retail shelf displays match their intended layout.

## Features
- Compare uploaded shelf images with reference planograms
- Detect empty spaces on shelves
- Provide similarity scores and analysis
- Easy-to-use interface with visual feedback

## How to Use
1. Upload an image of your shelf
2. The system will automatically compare it with the reference planogram
3. Get instant feedback on empty spaces and layout compliance

## Technical Details
- Built with Chainlit
- Uses CLIP for image comparison
- OpenCV for image processing
- PyTorch for deep learning operations

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/planolyzer.git
cd planolyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate your virtual environment if you haven't already:
```bash
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Run the application:
```bash
chainlit run app.py
```

## Project Structure

```
planolyzer/
├── requirements.txt    # Project dependencies
├── README.md          # This file
└── app.py            # Main application file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 