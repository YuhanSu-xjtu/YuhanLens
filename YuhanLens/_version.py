import json

version_json = '''
{
 "date": "2022-04-17T00:00:00-0000",
 "author": "YuhanSu",
 "version": "0.1.4"
}
'''


def get_versions():
    return json.loads(version_json)
