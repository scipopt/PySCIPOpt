"""Download a scipoptsuite-deploy zip to scip.zip, for the Windows wheels.

Usage: python scripts/fetch_scip.py <url of libscip-windows.zip>

SCIP_ARTIFACT_SUFFIX ("-debug", "-exact") is added to the file name, as the
Linux and macOS commands in pyproject.toml do. Windows needs this script
because its before-all runs in cmd, where an unset variable isn't empty.
"""

import os
import sys
import urllib.request

url = sys.argv[1].removesuffix(".zip") + os.environ.get("SCIP_ARTIFACT_SUFFIX", "") + ".zip"
print(f"Downloading {url}")
urllib.request.urlretrieve(url, "scip.zip")
