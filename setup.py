"""
setup.py — Package installation for vsp-3d-reconstruction.

Install in development mode: pip install -e .
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    # Strip comments and blank lines
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="vsp-3d-reconstruction",
    version="2.0.0",
    author="VSP-3D Contributors",
    author_email="vsp3d@example.com",
    description=(
        "AI-powered virtual surgical planning and 3D reconstruction "
        "from CT/CBCT for CMF and orthopedic surgery"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vsp-3d-reconstruction",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/vsp-3d-reconstruction/issues",
        "Documentation": "https://github.com/yourusername/vsp-3d-reconstruction/docs/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where=".", include=["src*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.1",
            "mypy>=1.6",
            "pre-commit>=3.4",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipywidgets>=8.1",
            "itkwidgets>=0.32",
        ],
        "full": [
            "pyvista>=0.43",
            "vtk>=9.2",
            "open3d>=0.17",
            "monai>=1.3",
            "einops>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "vsp-segment=scripts.segment:main",
            "vsp-reconstruct=scripts.reconstruct:main",
            "vsp-plan=scripts.plan_surgery:main",
            "vsp-export=scripts.export_stl:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "docs/*.md"],
    },
)
