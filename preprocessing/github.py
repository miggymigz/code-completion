import os
import requests


API_BASE_URL = 'https://api.github.com'
LATEST_RELEASE_API = '/repos/{}/{}/releases/latest'


def get_latest_release_tarball_url(user, name):
    latest_release_api = LATEST_RELEASE_API.format(user, name)
    api_url = API_BASE_URL + latest_release_api
    r = requests.get(api_url, headers=get_header_for_auth())

    # Ensure API requests return OK
    if r.status_code != 200:
        raise AssertionError(r.text)

    # return download link for the source zip
    return r.json()['tarball_url']


def download_latest_release(user, name, path):
    url = get_latest_release_tarball_url(user, name)
    r = requests.get(url, stream=True)

    with open(path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


def get_header_for_auth():
    # retrieve access token from environment vars
    if 'GITHUB_ACCESS_TOKEN' not in os.environ:
        raise EnvironmentError(
            'Error: "GITHUB_ACCESS_TOKEN" variable not found!')

    access_token = os.environ['GITHUB_ACCESS_TOKEN']
    auth_value = 'token {}'.format(access_token)

    return {'Authorization': auth_value}
