"""Setup script for dicom_qc package."""

from setuptools import setup, find_packages

# Read requirements
requirements = [
    'xnat>=0.4.0',
    'pydicom>=2.3.0',
    'numpy>=1.20.0',
    'matplotlib>=3.5.0',
    'pillow>=9.0.0',
    'ipywidgets>=8.0.0',
    'jinja2>=3.0.0',
    'ipython>=7.0.0',
]

setup(
    name='dicom_qc',
    version='0.1.0',
    description='DICOM QC Review Tool for post-deidentification quality checks',
    author='Kate Alpert',
    author_email='kate@embarklabs.ai',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'dicom_qc': ['reports/templates/*.html'],
    },
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    keywords='dicom, xnat, qc, quality control, medical imaging',
)
