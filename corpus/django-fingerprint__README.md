# Django Usage Statistics

**Goal**: Demonstrate Django's widespread adoption among high-traffic websites to support Django Software Foundation fundraising.

**Unique Value**: Real-time detection on the world's most popular websites with live popularity rankings.

## Credit to Thibaud Colas
This project was 100% inspired by the work of [thibaud](https://github.com/thibaudcolas) and his amazing blog post [Django in Government](https://thib.me/django-in-government). Thank you, Thibaud!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/django-fingerprint.git
cd django-fingerprint

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Using uv (faster alternative):**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quick Start

```bash
# Run big scan (automatically shows comprehensive analysis when done)
python scan.py --limit 100000
```

**Output includes:**
- Total sites scanned and Django detection rate
- Detection methods breakdown (session cookies, admin CSS, CSRF tokens, etc.)
- **Notable high-profile sites with RANKINGS** (Rank #33, #185, etc.)
- High-confidence (80%+) Django sites

## Latest Results

**Last Updated:** 2025-12-25 18:48:25

### Summary

- **Total sites scanned:** 100,000
- **Django sites found:** 308
- **Detection rate:** 0.3%

### Top Django Versions

| Version | Count | Percentage |
|---------|-------|------------|
| Unknown | 308 | 100.0% |

### Detected Django Sites

*Showing top 50 of 308 Django sites detected*

| Rank | Site | Confidence | Version | Detection Method | MD5 Hash | Admin Path |
|------|------|-----------|---------|------------------|----------|------------|
| 18 | linkedin.com | 85% | Unknown | sessionid cookie | - | - |
| 46 | youtu.be | 80% | Unknown | Admin CSS | `69b62ebf...` | `/static/admin/css/base.css` |
| 840 | agoda.com | 85% | Unknown | sessionid cookie | - | - |
| 924 | chaturbate.com | 75% | Unknown | csrftoken cookie, CSRF token | - | - |
| 1,117 | coursera.org | 80% | Unknown | Admin CSS | `49dc31b5...` | `/admin/css/base.css` |
| 1,370 | suumo.jp | 70% | Unknown | sessionid cookie | - | - |
| 2,113 | bytedance.com | 80% | Unknown | Admin CSS | `a657d394...` | `/django/admin/css/base.css` |
| 2,199 | semrush.com | 70% | Unknown | CSRF token | - | - |
| 2,628 | parallels.com | 60% | Unknown | CSRF token | - | - |
| 2,719 | 5movierulz.dental | 80% | Unknown | Admin CSS | `d7897be2...` | `/django-admin/css/base.css` |
| 3,405 | 5movierulz.gripe | 80% | Unknown | Admin CSS | `1d2a72e5...` | `/django/admin/css/base.css` |
| 3,955 | emojipedia.org | 85% | Unknown | sessionid cookie | - | - |
| 4,069 | forticloud.com | 85% | Unknown | sessionid cookie | - | - |
| 4,324 | pinterest.co.uk | 75% | Unknown | csrftoken cookie | - | - |
| 4,457 | pin.it | 80% | Unknown | Admin CSS | `fcd1cdd2...` | `/clientadmin/css/base.css` |
| 4,880 | etherscan.io | 85% | Unknown | sessionid cookie | - | - |
| 5,481 | band.us | 85% | Unknown | sessionid cookie | - | - |
| 5,938 | tiny.cc | 80% | Unknown | Admin CSS | `3f4f316d...` | `/assets/admin/css/base.css` |
| 7,345 | toyota.jp | 70% | Unknown | sessionid cookie | - | - |
| 7,441 | gameforge.com | 80% | Unknown | Admin CSS | `5dd5344f...` | `/django/admin/css/base.css` |
| 7,801 | licindia.in | 85% | Unknown | sessionid cookie | - | - |
| 8,153 | pinterest.de | 80% | Unknown | Admin CSS | `d5a5f53b...` | `/django-admin/css/base.css` |
| 8,185 | fikfap.com | 80% | Unknown | Admin CSS | `b865b3d6...` | `/staticfiles/admin/css/base.css` |
| 8,336 | mgeko.cc | 75% | Unknown | csrftoken cookie, CSRF token | - | - |
| 8,691 | elong.com | 80% | Unknown | Admin CSS | `80b29199...` | `/django/admin/css/base.css` |
| 9,576 | scba.gov.ar | 70% | Unknown | sessionid cookie | - | - |
| 9,656 | teachoo.com | 80% | Unknown | Admin CSS | `c52f7bea...` | `/static/admin/css/base.css` |
| 9,748 | workman.jp | 70% | Unknown | sessionid cookie | - | - |
| 9,986 | goneo.de | 50% | Unknown | csrftoken cookie | - | - |
| 10,096 | can.az | 80% | Unknown | Admin CSS | `9c88aca4...` | `/django/admin/css/base.css` |
| 10,483 | reasonsecurity.com | 80% | Unknown | Admin CSS | `b9c64d39...` | `/django/admin/css/base.css` |
| 10,549 | bscscan.com | 85% | Unknown | sessionid cookie | - | - |
| 10,847 | 1024tera.com | 75% | Unknown | csrftoken cookie | - | - |
| 11,082 | nv.gov | 70% | Unknown | sessionid cookie | - | - |
| 11,620 | ihk.de | 85% | Unknown | sessionid cookie | - | - |
| 11,947 | tudou.com | 80% | Unknown | Admin CSS | `5f56459d...` | `/staticfiles/admin/css/base.css` |
| 11,987 | coolors.co | 80% | Unknown | Admin CSS | `3bae42a2...` | `/django/admin/css/base.css` |
| 12,236 | lung.org | 70% | Unknown | sessionid cookie | - | - |
| 13,083 | szarada.net | 75% | Unknown | csrftoken cookie | - | - |
| 13,145 | kone.com | 70% | Unknown | sessionid cookie | - | - |
| 13,218 | torontosun.com | 70% | Unknown | CSRF token | - | - |
| 13,417 | tsite.jp | 70% | Unknown | sessionid cookie | - | - |
| 13,419 | ticketnew.com | 80% | Unknown | Admin CSS | `4e4a33e5...` | `/django-admin/css/base.css` |
| 13,475 | openwrt.org | 80% | Unknown | Admin CSS | `1bcf3f0d...` | `/static/admin/css/base.css` |
| 13,827 | classera.com | 80% | Unknown | Admin CSS | `d2030db1...` | `/static/admin/css/base.css` |
| 13,838 | freesound.org | 75% | Unknown | csrftoken cookie, CSRF token | - | - |
| 13,863 | cinemark.com | 85% | Unknown | sessionid cookie | - | - |
| 13,881 | foreflight.com | 75% | Unknown | csrftoken cookie, CSRF token | - | - |
| 13,951 | oas.org | 70% | Unknown | sessionid cookie | - | - |
| 13,955 | antenna.gr | 70% | Unknown | sessionid cookie | - | - |


## Output Files

```
data/results/
├── detection_results.json          # Raw detection data (all 100K sites)
├── summary.txt                     # Comprehensive analysis (shown above)
├── analysis_report.txt             # Detailed report
├── analysis_results.csv            # Full spreadsheet export (all 100K sites)
└── django_sites_fundraising.csv    # Django-detected sites only, all columns
```

### Fundraising CSV

The fundraising CSV (`django_sites_fundraising.csv`) contains only the 308 Django-detected sites, sorted by Tranco rank (most popular first), with 15 columns:

| Column | Description |
|--------|-------------|
| Tranco Rank | Global traffic rank — lower = more popular |
| URL | Full URL |
| Domain | Clean domain name |
| Confidence % | Numeric score (60–85) for sorting/filtering |
| Confidence Level | High (≥80%) / Medium (60–79%) / Low (<60%) |
| Detection Methods | Human-readable signals found |
| Signal Count | Number of independent signals |
| Session Cookie | Django `sessionid` cookie detected |
| CSRF Signal | CSRF token or cookie detected |
| Admin CSS | Django admin CSS fingerprint matched |
| Admin CSS Path | Exact URL of the admin CSS file |
| Django Language Cookie | `django_language` cookie detected |
| X-Frame-Options Header | Security header present |
| X-Content-Type-Options Header | Security header present |
| Detected Version | Django version (pending version detection) |

Regenerate it after a new scan:

```bash
python scripts/generate_fundraising_csv.py
```

## Detection Methods

All methods implemented and tested:
- ✓ Admin CSS MD5 fingerprints (9 paths, 60 matches)
- ✓ Session/CSRF cookies (175 matches)
- ✓ JavaScript variables (__admin_media_prefix__, window.django)
- ✓ Django REST Framework API detection
- ✓ Django debug pages

## Options

### Basic Usage

```bash
# Default scan (uses Tranco list, resumes from cache)
python scan.py --limit 100000

# Continue a previous scan (just increase the limit - it will use cached results)
python scan.py --limit 200000  # Scans 100K more, keeps previous 100K

# Maximum limit: 1,000,000 sites (full Tranco top 1M list)
python scan.py --limit 1000000
```

### Data Source

Scans use the **Tranco top 1M list** - a research-grade ranking of the most popular websites updated daily.

```bash
# Scan top 100K sites
python scan.py --limit 100000

# Scan up to 1M sites
python scan.py --limit 1000000
```

### Cache Control

```bash
# Resume from cache (default - RECOMMENDED)
python scan.py --limit 100000

# Start completely fresh (WARNING: discards all cached results!)
python scan.py --fresh --limit 100000

# Cache is valid for 7 days by default
# After 7 days, scan automatically starts fresh
```

### Analysis Only

```bash
# Re-run analysis without scanning
python scripts/comprehensive_analysis.py

# Regenerate the fundraising CSV (e.g. after a new scan)
python scripts/generate_fundraising_csv.py
```

**Important Notes:**
- **Incremental saves:** Progress is saved every 1,000 sites
- **Resume capability:** If interrupted, just re-run with same limit to continue
- **Cache behavior:** By default, scans use cached results to avoid re-scanning
- **Don't use --fresh to continue:** Use --fresh only to discard all previous results

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (48 passing)
pytest tests/ -v

# Run linters
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Project Structure
```
scan.py                             # Main entry point
scripts/
├── comprehensive_analysis.py       # Best results viewer
├── detect_sites.py                 # Core detection
├── generate_fundraising_csv.py     # Fundraising CSV export
└── run_scan.py                     # Pipeline orchestration
src/django_fingerprint/
├── async_detector.py               # Production detector
├── detector.py                     # Legacy (tested, not used)
└── analyzer.py                     # Result analysis
tests/                              # 48 tests passing
```

## Roadmap for DSF Fundraising Impact

### High Priority (Immediate Impact)

1. **Version Detection** - Currently 100% "Unknown". Implement Django version fingerprinting via:
   - Server headers (X-Django-Version if exposed)
   - Admin static file hashes across Django versions
   - Error page signatures
   - *Impact: "308 sites running Django 4.2+" is more compelling than "Unknown"*

2. **Notable Sites Curation** - Only 1 site flagged as "notable" (rank #924). Add:
   - Industry categorization (fintech, e-commerce, media, government, education)
   - Traffic estimates from Tranco data
   - Highlight Fortune 500, government sites, major brands
   - *Impact: "Django powers 15 Fortune 500 companies" resonates with donors*

3. **False Positive Reduction** - 70% confidence threshold may include false positives:
   - Validate sessionid cookie detections (too generic)
   - Require multiple signals for <80% confidence
   - Manual verification of top 50 sites
   - *Impact: Credibility is critical for fundraising materials*

### Medium Priority (Enhanced Messaging)

4. **Traffic Impact Metrics** - Convert rankings to estimated monthly visitors:
   - Tranco rank #18 (LinkedIn) → ~1B monthly visits
   - Aggregate "Django serves X billion pageviews/month"
   - *Impact: Scale matters more than site count*

5. **Geographic & Industry Analysis** - Breakdown by:
   - Country TLDs (.jp, .de, .in showing strong presence)
   - Industry sectors (fintech: etherscan.io, travel: agoda.com)
   - *Impact: "Django dominates fintech in Asia" tells a story*

6. **Trend Tracking** - Store historical scan data:
   - Month-over-month Django adoption changes
   - New high-profile sites detected
   - Version migration patterns
   - *Impact: "Django adoption grew 12% this quarter" shows momentum*

### Lower Priority (Technical Improvements)

7. **Detection Method Weights** - Admin CSS (80%) more reliable than sessionid (70%):
   - Confidence scoring should reflect detection method quality
   - Combine multiple weak signals into strong detection

8. **Performance Optimization** - Scale to full 1M Tranco list:
   - Current: 100K sites in ~X hours
   - Target: 1M sites overnight for comprehensive data

9. **Export Formats** - Generate fundraising-ready materials:
   - PDF report with charts
   - Social media graphics (top sites infographic)
   - Press release template with key statistics

### Data Quality Notes

- **Suspicious sites** (5movierulz.dental, fikfap.com) may hurt credibility - consider filtering
- **LinkedIn (rank #18)** is highest-profile detection - verify manually and highlight prominently
- **Government sites** (nv.gov, scba.gov.ar, oas.org) excellent for credibility
- **Education** (coursera.org) and **open source** (openwrt.org) align with Django's mission

## License

MIT
