import json

version_json = '''
{
 "date": "2022-04-22T00:00:00-0000",
 "author": "Yuhan Su",
 "version": "0.1.7",
 "organization":"Xi'an Jiaotong University"
}
'''


def get_versions():
    return json.loads(version_json)
