"""Test report generation"""

import pytest

def test_health_report_generator():
    try:
        from REPORTS.health_report_generator import HealthReportGenerator
        generator = HealthReportGenerator(output_dir="TESTS/temp_reports")
        summary = generator.generate_quick_summary()
        
        assert isinstance(summary, dict)
        assert 'timestamp' in summary
    except ImportError:
        pytest.skip("HealthReportGenerator not available")
