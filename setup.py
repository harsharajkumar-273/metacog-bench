from setuptools import setup, find_packages

setup(
    name="metacog-bench",
    version="1.0.0",
    description="A Metacognitive Probing & Calibration Evaluation Suite for Frontier LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Applied Cognitive Evaluation Group",
    url="https://github.com/your-username/metacog-bench",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.0",
        "streamlit>=1.30.0",
        "plotly>=5.0.0",
        "pydantic>=2.0.0"
    ],
)
