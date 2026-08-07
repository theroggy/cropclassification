"""Package with functionalities to support agricultural parcel monitoring."""

import logging

# Disable info logging pf botocore.credentials
logger_botocore_credentials = logging.getLogger("botocore.credentials")
logger_botocore_credentials.setLevel(logging.WARNING)
