"""Firebase project constants for the jobs system (admin + user editions).

This is the CLIENT web config — it is public by design (the same values
ship inside momo-imagine's index.html). Access control lives in the
Firestore security rules, NOT in the secrecy of these values. Never add
a service-account key or any AI API key to this file: the user edition
repo is published publicly on GitHub.
"""

FIREBASE_PROJECT_ID = "momo-imagine"
FIREBASE_WEB_API_KEY = "AIzaSyBRsWgZrlvqW-20wm1N8tcyeJnD5K0rJSw"

# Firestore collection holding one doc per job; mscript payloads live in
# the `payload` subcollection of each job doc (gzip+base64, chunked).
JOBS_COLLECTION = "script_jobs"

FIRESTORE_BASE = (
    "https://firestore.googleapis.com/v1/projects/"
    f"{FIREBASE_PROJECT_ID}/databases/(default)/documents"
)
