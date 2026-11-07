import json
import os
import re
from datetime import datetime, timezone

import httpx
from openai import OpenAI

import config

REDACTED = '[REDACTED]'
# Error fields the API fills from a fixed vocabulary: the error type, the error code and the
# name of the offending parameter. They do not carry the key and are passed on unchanged.
PROTOCOL_FIELDS = ('type', 'code', 'param')
# The SDK re-sends a request that met 429 or 5xx, with backoff, this many times (default 2),
# so parallel images do not turn rate limits into failures (CHECKLIST.md C9). Every
# response, 429s included, is logged.
MAX_RETRIES = 6


def redact(text, api_key):
    """Replace echoes of the key in free text: the key itself and tokens shaped like it.

    Such a token is masked like the 401 message quotes the key (head****tail), is an sk- key,
    or starts with the key's first 8 characters and ends with its last 4. A fragment on its
    own is left alone: the last 4 characters may be part of any word, such as rate or _key.
    """
    text = text.replace(api_key, REDACTED)
    text = re.sub(r"[^\s'\"]*\*{3,}[^\s'\".,;]*", REDACTED, text)
    text = re.sub(r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}', REDACTED, text)
    head, tail = re.escape(api_key[:8]), re.escape(api_key[-4:])
    return re.sub(rf"(?<![A-Za-z0-9_-]){head}[^\s'\"]*{tail}(?![A-Za-z0-9_-])", REDACTED, text)


def redact_values(value, api_key):
    """redact() every string in a parsed JSON body, except the values of PROTOCOL_FIELDS."""
    if isinstance(value, str):
        return redact(value, api_key)
    if isinstance(value, dict):
        return {k: v if k in PROTOCOL_FIELDS and isinstance(v, str) else redact_values(v, api_key)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_values(v, api_key) for v in value]
    return value


def redact_response(response, api_key):
    """A copy of an error response without echoes of the key, or the response itself if it has none.

    A JSON body is redacted value by value and serialized again, so the copy still parses.
    """
    try:
        text = response.text
    except httpx.ResponseNotRead:  # closed unread: the SDK reports the status code only
        return response
    try:
        body = json.loads(text)
    except ValueError:
        redacted = redact(text, api_key)
        if redacted == text:
            return response
    else:
        redacted_body = redact_values(body, api_key)
        if redacted_body == body:
            return response
        redacted = json.dumps(redacted_body)
    headers = [(k, v) for k, v in response.headers.items()
               if k.lower() not in ('content-encoding', 'content-length')]
    return httpx.Response(response.status_code, headers=headers, request=response.request,
                          content=redacted.encode())


class RedactedOpenAI(OpenAI):
    """OpenAI client whose API errors carry no echo of the key.

    The SDK builds every status error (401, 429, ...) from the response here, so it is handed
    the redacted response: str(e), which the scripts write to their logs, prints and result
    CSVs, e.body and e.response then all come from the same clean content. The status code,
    error type, error code and param stay; the request id is in e.response.headers and in
    API_LOG_PATH.
    """

    def _make_status_error_from_response(self, response):
        return super()._make_status_error_from_response(redact_response(response, self.api_key))


def log_response(response):
    """httpx response hook: append one JSON line per API response to config.API_LOG_PATH.

    Only the fields below are written. Request headers (which carry the key), message
    contents and error messages (401 messages quote part of the key) are not.
    """
    response.read()
    try:
        request = json.loads(response.request.content)
    except ValueError:
        request = {}
    try:
        body = response.json()
    except ValueError:
        body = {}
    record = {
        'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'endpoint': response.request.url.path,
        'status': response.status_code,
        'request_id': response.headers.get('x-request-id'),
        'model_requested': request.get('model'),
        'model_served': body.get('model'),
        'temperature': request.get('temperature'),
        'max_tokens': request.get('max_tokens'),
        'max_completion_tokens': request.get('max_completion_tokens'),
        'reasoning_effort': request.get('reasoning_effort'),
        'service_tier': body.get('service_tier'),
        'finish_reasons': [c.get('finish_reason') for c in body.get('choices', [])],
        'empty_contents': [not bool((c.get('message') or {}).get('content')) for c in body.get('choices', [])],
        'usage': body.get('usage'),
        'error_code': (body.get('error') or {}).get('code'),
        'error_param': (body.get('error') or {}).get('param'),
    }
    os.makedirs(os.path.dirname(config.API_LOG_PATH), exist_ok=True)
    with open(config.API_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def openai_client(transport=None):
    """OpenAI client (key from .env, official endpoint) whose responses are logged."""
    http_client = httpx.Client(transport=transport, follow_redirects=True,
                               event_hooks={'response': [log_response]})
    return RedactedOpenAI(api_key=config.load_api_key(), base_url=config.OPENAI_API_BASE,
                          http_client=http_client, max_retries=MAX_RETRIES)
