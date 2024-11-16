from setuptools import setup, find_packages

setup(
    name="retail-abm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "agent-torch",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "run-simulation=scripts.run_simulation:main",
        ],
    },
)
