# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime

def generate_session_id() -> str:
    """Generate a unique session ID based on current timestamp."""
    current_datetime = datetime.datetime.now()
    date_string = current_datetime.strftime("%Y%m%d%H%M%S")
    return date_string
