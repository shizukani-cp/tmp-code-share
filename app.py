from dotenv import load_dotenv
import os, sys, argparse, json
import requests

load_dotenv()

ENCODE = (os.environ.get("LANG") or "ja-JP.UTF-8").split(".")[1]

class CloudflareWorkersClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def get(self, path):
        response = requests.get(f'{self.base_url}{path}', headers=self.headers)
        if response.status_code == 200:
            return response.json() if path.endswith('/') else response.text
        else:
            return {"error": responce.status_code, "text": response.text}

    def post(self, path, content=None):
        data = content if content else ''
        response = requests.post(f'{self.base_url}{path}', headers=self.headers, data=data)
        return {"status": response.status_code, "text": response.text}

    def put(self, path, content):
        response = requests.put(f'{self.base_url}{path}', headers=self.headers, data=content)
        return {"status": response.status_code, "text": response.text}

    def delete(self, path):
        response = requests.delete(f'{self.base_url}{path}', headers=self.headers)
        return {"status": response.status_code, "text": response.text}

def gen_create_dir_handler(client):
    def out(args):
        responce = client.post(args.path)
        if args.prntrspnc:
            print(responce)
    return out

def gen_create_file_handler(client):
    def out(args):
        responce = client.post(args.path, args.contents.read())
        if args.prntrspnc:
            print(responce)
    return out

def gen_get_handler(client):
    def out(args):
        print(client.get(args.path))
    return out

def gen_put_handler(client):
    def out(args):
        responce = client.put(args.path, args.contents.read())
        if args.prntrspnc:
            print(responce)
    return out

def gen_delete_handler(client):
    def out(args):
        responce = client.delete(args.path)
        if args.prntrspnc:
            print(responce)
    return out

if __name__ == '__main__':
    base_url = os.environ.get("TMP_FILE_SHARE_BASE_URL")
    api_key = os.environ.get("TMP_FILE_SHARE_API_KEY")

    client = CloudflareWorkersClient(base_url, api_key)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prntrspnc", "-p", action="store_true")
    subparsers = parser.add_subparsers()

    parser_create = subparsers.add_parser("create")
    subparser_create = parser_create.add_subparsers()

    parser_create_dir = subparser_create.add_parser("dir")
    parser_create_dir.add_argument("path")
    parser_create_dir.set_defaults(handler=gen_create_dir_handler(client))

    parser_create_file = subparser_create.add_parser("file")
    parser_create_file.add_argument("path")
    parser_create_file.add_argument("--contents", "-c",
                                    type=argparse.FileType("r", encoding=ENCODE),
                                    default=sys.stdin)
    parser_create_file.set_defaults(handler=gen_create_file_handler(client))

    parser_get = subparsers.add_parser("get")
    parser_get.add_argument("path")
    parser_get.set_defaults(handler=gen_get_handler(client))

    parser_put = subparsers.add_parser("put")
    parser_put.add_argument("path")
    parser_put.add_argument("--contents", "-c",
                            type=argparse.FileType("r", encoding=ENCODE),
                            default=sys.stdin)
    parser_put.set_defaults(handler=gen_put_handler(client))

    parser_delete = subparsers.add_parser("delete")
    parser_delete.add_argument("path")
    parser_delete.set_defaults(handler=gen_delete_handler(client))

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()

