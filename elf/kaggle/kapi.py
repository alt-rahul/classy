"""Minimal Kaggle REST client (plain curl; the session proxy injects the user's credential).

Never add an auth header here and never print or store credentials. Everything this creates is private.
"""
import json, os, subprocess, tempfile, time

API = 'https://www.kaggle.com/api/v1'
OWNER = 'ograhulrajkumar'


class KaggleError(RuntimeError):
    pass


def _curl(args, timeout=120):
    r = subprocess.run(['curl', '-s', '--max-time', str(timeout), '-w', '\n%{http_code}'] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise KaggleError('curl exit %d: %s' % (r.returncode, r.stderr.strip()[-300:]))
    body, _, code = r.stdout.rpartition('\n')
    return int(code or 0), body


def call(method, path, body=None, timeout=120, ok=(200,)):
    args = ['-X', method]
    if body is not None:
        args += ['-H', 'Content-Type: application/json', '--data-binary', '@-']
    url = API + path
    if body is not None:
        r = subprocess.run(['curl', '-s', '--max-time', str(timeout), '-w', '\n%{http_code}'] + args + [url],
                           input=json.dumps(body), capture_output=True, text=True)
        if r.returncode != 0:
            raise KaggleError('curl exit %d: %s' % (r.returncode, r.stderr.strip()[-300:]))
        txt, _, code = r.stdout.rpartition('\n')
        code = int(code or 0)
    else:
        code, txt = _curl(args + [url], timeout)
    if code not in ok:
        raise KaggleError('%s %s -> HTTP %d: %s' % (method, path, code, txt[:500]))
    try:
        return json.loads(txt) if txt.strip() else {}
    except ValueError:
        raise KaggleError('%s %s -> non-JSON: %s' % (method, path, txt[:300]))


def upload_blob(path, kind='DATASET'):
    """Upload one file through the blob flow; returns the blob token."""
    st = os.stat(path)
    d = call('POST', '/blobs/upload', {'type': kind, 'name': os.path.basename(path),
                                       'contentLength': st.st_size, 'lastModifiedEpochSeconds': int(st.st_mtime)})
    url, tok = d.get('createUrl'), d.get('token')
    if not url or not tok:
        raise KaggleError('blob upload start failed: %s' % list(d.keys()))
    for attempt in range(3):
        code, _ = _curl(['-X', 'PUT', '-H', 'Content-Type: application/octet-stream', '--data-binary', '@' + path, url],
                        timeout=600)
        if code in (200, 201):
            return tok
        time.sleep(5)
    raise KaggleError('blob PUT failed for %s (HTTP %d)' % (path, code))


def dataset_status(slug):
    try:
        d = call('GET', '/datasets/status/%s/%s' % (OWNER, slug))
    except KaggleError as e:
        if 'HTTP 404' in str(e) or 'HTTP 403' in str(e):
            return None
        raise
    return d.get('status', d) if isinstance(d, dict) else d


def dataset_create(slug, title, paths, license_name='Apache 2.0'):
    files = [{'token': upload_blob(p)} for p in paths]
    body = {'title': title, 'slug': slug, 'ownerSlug': OWNER, 'licenseName': license_name, 'isPrivate': True,
            'convertToCsv': False, 'categoryIds': [], 'files': files}
    return call('POST', '/datasets/create/new', body, timeout=300)


def dataset_version(slug, paths, notes):
    files = [{'token': upload_blob(p)} for p in paths]
    body = {'versionNotes': notes, 'files': files, 'convertToCsv': False, 'categoryIds': [], 'deleteOldVersions': False}
    return call('POST', '/datasets/create/version/%s/%s' % (OWNER, slug), body, timeout=300)


def kernel_push(slug, title, text, datasets=(), competitions=('santa-2026',), kernels=()):
    body = {'slug': '%s/%s' % (OWNER, slug), 'newTitle': title, 'text': text, 'language': 'python',
            'kernelType': 'script', 'isPrivate': True, 'enableGpu': False, 'enableTpu': False,
            'enableInternet': False, 'datasetDataSources': list(datasets),
            'competitionDataSources': list(competitions), 'kernelDataSources': list(kernels),
            'modelDataSources': [], 'categoryIds': []}
    return call('POST', '/kernels/push', body, timeout=300)


def kernel_status(slug):
    return call('GET', '/kernels/status?userName=%s&kernelSlug=%s' % (OWNER, slug))


def kernel_output(slug, dest):
    """Download a finished kernel's output files into dest; returns (list of file names, log text)."""
    os.makedirs(dest, exist_ok=True)
    names, log, token = [], '', None
    for _ in range(20):
        q = '/kernels/output?userName=%s&kernelSlug=%s' % (OWNER, slug) + ('&pageToken=%s' % token if token else '')
        d = call('GET', q, timeout=300)
        log = d.get('log') or log
        for f in d.get('files', []):
            name = os.path.basename(f.get('fileName', ''))
            if not name or not f.get('url'):
                continue
            tmp = os.path.join(dest, '.' + name + '.part')
            code, _ = _curl(['-L', '-o', tmp, f['url']], timeout=600)
            if code != 200:
                raise KaggleError('download %s failed HTTP %d' % (name, code))
            os.replace(tmp, os.path.join(dest, name))
            names.append(name)
        token = d.get('nextPageToken')
        if not token:
            break
    if log:
        with open(os.path.join(dest, 'kernel_log.json'), 'w') as f:
            f.write(log if isinstance(log, str) else json.dumps(log))
    return names, log


def list_my_kernels(search=''):
    return call('GET', '/kernels/list?user=%s&pageSize=100%s' % (OWNER, ('&search=' + search) if search else ''))
