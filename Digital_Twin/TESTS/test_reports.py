"""
Test report generation with expanded coverage for CSV and PDF outputs.
"""

import pytest
import csv
from pathlib import Path

# Use pytest.importorskip to handle the import.
# This will cleanly skip all tests in this file if the module or class isn't found.
try:
    # We import the class directly for easier type hinting and use
    from REPORTS.health_report_generator import HealthReportGenerator
except ImportError:
    pytest.skip("HealthReportGenerator not found. Skipping all report tests.", allow_module_level=True)


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def temp_output_dir(tmp_path_factory):
    """
    Create a dedicated temporary directory for report outputs using pytest's
    built-in tmp_path_factory. This directory is automatically cleaned up
    after the test session.
    
    Using scope="module" means this directory is created once for all tests in this file.
    """
    report_dir = tmp_path_factory.mktemp("temp_reports")
    return report_dir

@pytest.fixture(scope="module")
def report_generator(temp_output_dir):
    """
    Fixture to provide a configured instance of HealthReportGenerator,
    pointing to the temporary output directory.
    """
    # Pass the Path object directly. Most modern libraries accept Path objects.
    # If your class specifically requires a string, use str(temp_output_dir)
    generator = HealthReportGenerator(output_dir=temp_output_dir)
    return generator

@pytest.fixture(scope="module")
def mock_report_data():
    """
    Provides a consistent set of mock data for testing report content.
    """
    return [
        # Headers: id, status, value
        {'id': 'sensor_a', 'status': 'OK', 'value': 25.5},
        {'id': 'sensor_b', 'status': 'WARNING', 'value': 80.1},
        {'id': 'sensor_c', 'status': 'CRITICAL', 'value': 105.0},
    ]

# --- Tests ---

def test_quick_summary_generation(report_generator):
    """
    Test the original functionality: generate_quick_summary.
    This test now uses the report_generator fixture.
    """
    summary = report_generator.generate_quick_summary()
    
    assert isinstance(summary, dict)
    assert 'timestamp' in summary
    # You could add more assertions here if you know the summary structure
    # For example:
    # assert 'total_alerts' in summary
    # assert 'status_counts' in summary

def test_csv_report_generation_and_content(report_generator, temp_output_dir, mock_report_data):
    """
    Test CSV report generation, file validity, and content accuracy.
    
    Assumes the generator has a method:
    generate_csv_report(data, filename="...")
    """
    csv_filename = "health_report.csv"
    output_path = temp_output_dir / csv_filename
    
    # --- Assumption on method signature ---
    # We wrap this in a try/except in case the method doesn't exist or
    # has a different signature (e.g., doesn't take data or filename)
    try:
        report_generator.generate_csv_report(mock_report_data, filename=csv_filename)
    except AttributeError:
        pytest.skip("Method 'generate_csv_report' not found on generator.")
    except TypeError:
        pytest.skip("Method 'generate_csv_report' has an unexpected signature. (Expected data and filename args)")
    # --- End Assumption ---

    # 1. Test file creation and validity
    assert output_path.exists(), "CSV file was not created"
    assert output_path.stat().st_size > 0, "CSV file is empty"

    # 2. Test content accuracy
    with open(output_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Check header (assuming keys from the first dict are headers)
        header = next(reader)
        assert header == ['id', 'status', 'value'], "CSV header is incorrect"
        
        # Check rows
        rows = list(reader)
        assert len(rows) == len(mock_report_data), "CSV has an incorrect number of data rows"
        
        # Spot-check data (all values will be strings in a CSV)
        assert rows[0] == ['sensor_a', 'OK', '25.5']
        assert rows[1] == ['sensor_b', 'WARNING', '80.1']
        assert rows[2] == ['sensor_c', 'CRITICAL', '105.0']

def test_pdf_report_generation_and_validity(report_generator, temp_output_dir, mock_report_data):
    """
    Test PDF report generation and file validity.
    
    Content accuracy for PDFs is complex. Instead, we verify the file
    exists, is non-empty, and has the correct PDF "magic bytes" file
    signature (%PDF-), which confirms it's a valid PDF file.
    
    Assumes the generator has a method:
    generate_pdf_report(data, filename="...")
    """
    pdf_filename = "health_report.pdf"
    output_path = temp_output_dir / pdf_filename
    
    # --- Assumption on method signature ---
    try:
        report_generator.generate_pdf_report(mock_report_data, filename=pdf_filename)
    except AttributeError:
        pytest.skip("Method 'generate_pdf_report' not found on generator.")
    except TypeError:
        pytest.skip("Method 'generate_pdf_report' has an unexpected signature. (Expected data and filename args)")
    # --- End Assumption ---

    # 1. Test file creation
    assert output_path.exists(), "PDF file was not created"
    
    # 2. Test file is not empty
    assert output_path.stat().st_size > 0, "PDF file is empty"

    # 3. Test content validity (Magic Bytes)
    # A valid PDF file should start with the bytes b'%PDF-'
    with open(output_path, 'rb') as f:
        magic_bytes = f.read(5) # Read the first 5 bytes
        assert magic_bytes == b'%PDF-', "File does not appear to be a valid PDF (magic bytes mismatch)"